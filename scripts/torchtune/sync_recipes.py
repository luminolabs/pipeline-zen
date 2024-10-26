from typing import List, Tuple

import argparse
import os
import re


# Path to Recipe Configs relative to Torchtune repository.
TORCHTUNE_RECIPE_CONFIG_SUFFIX = 'recipes/configs'

# Models whose recipe configs are to be converted to Lumino equivalents.
TORCHTUNE_RECIPE_MODELS = [
    'llama3_1',
    # 'llama3_2',
    # 'mistral'
]
# List of fields to remove from the torchtune recipe config.
FIELDS_TO_REMOVE = [
    'batch_size',
    'compile',
    'max_steps_per_epoch',
    'epochs',
    'seed',
    'shuffle',
]

BLOCK_NAMES_TO_REMOVE = ['metric_logger', 'profiler']

# Lines to be added to the Lumino config
LUMINO_CONFIG_HEADER = '''
# These are set by the job launcher, don't set them manually
base_model_path:
output_dir:
shuffle:
batch_size:
epochs:
lr:
seed:
'''

def _is_field_to_remove(line: str) -> bool:
    """Returns whether the line contains a field matching any of `FIELDS_TO_REMOVE`."""
    line = line.strip()
    for f in FIELDS_TO_REMOVE:
        if line.startswith(f):
            return False
    return True


def _next_block(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Given a list of lines return the next block and the remaining lines.

    A block is a continous list of lines with no empty line in between.
    However, empty lines which aren't followed by a line where the first
    character isn't whitespace are not counted as such.

    """
    ln = None
    found_empty_line = False
    for ln, line in enumerate(lines):
        if found_empty_line:
            if not line[0].isspace():
                # Found a line starting without space
                # after an empty line, so the block ends in the previous line.
                ln = ln - 1
                break
        if not line.strip():
            found_empty_line = True

    block, remaining = [], []
    if ln == len(lines) - 1:
        ln = ln + 1
    if ln:
        block, remaining = (lines[:ln], lines[ln+1:])
    return block, remaining


def _convert_config(src: str, dest: str):
    """Convert a Torchtune recipe from src to Lumino version in dest."""
    # Load the TT recipe
    src_lines: List[str] = []
    with open(src, 'r') as f:
        src_lines = f.readlines()
    if not src_lines:
        raise Exception(f'Unable to load: {src}')

    # Remove all fields as per func `_is_field_to_remove` 
    src_lines = list(filter(_is_field_to_remove, src_lines))

    dest_lines: List[str] = []
    block, src_lines = _next_block(src_lines)

    # First block is typically a large comment block from
    # which we retain just the first line.
    if block:
        dest_lines.append(block[0])
        dest_lines.append('\n')

    # Then we add the LUMINO_CONFIG_HEADER
    dest_lines.append(LUMINO_CONFIG_HEADER)
    dest_lines.append('\n')

    # Process all remaining blocks
    while True:
        block, src_lines = _next_block(src_lines)
        if not block and not src_lines:
            break

        block_names = ['tokenizer', 'dataset', 'checkpointer', 'optimizer']
        current_block_name = None
        for i in range(0, len(block)):
            if block[i].strip().startswith('#'):
                continue

            if not current_block_name:
                # Determine block name if we haven't already.
                for bn in block_names + BLOCK_NAMES_TO_REMOVE:
                    if block[i].startswith(bn):
                        current_block_name = bn
                        break

            line_parts = block[i].split(':')
            if current_block_name == 'tokenizer':
                # Change tokenizer -> path
                if line_parts[0].strip() == 'path':
                    tok_path = line_parts[1]
                    new_tok_path = re.sub('/tmp/[^/]+/', '${base_model_path}/', tok_path)
                    block[i] = ':'.join([line_parts[0], new_tok_path])

            if current_block_name == 'dataset':
                # Change dataset -> _component_
                if line_parts[0].strip() == '_component_':
                    block[i] = ': '.join([line_parts[0], ' # leave empty, this is injected by the job launcher'])
                    block[i] += '\n'

            if current_block_name == 'optimizer':
                # Change optimizer -> lr
                if line_parts[0].strip() == 'lr':
                    block[i] = ': '.join([line_parts[0], f'${{lr}}  #{line_parts[1]}'])
                    block[i] += '\n'

            if current_block_name == 'checkpointer':
                # Change checkpointer -> checkpoint_dir
                if line_parts[0].strip() == 'checkpoint_dir':
                    block[i] = ': '.join([line_parts[0], '${base_model_path}'])
                    block[i] += '\n'
                # Change checkpointer -> output_dir
                if line_parts[0].strip() == 'output_dir':
                    block[i] = ': '.join([line_parts[0], '${output_dir}'])
                    block[i] += '\n'

        if current_block_name in BLOCK_NAMES_TO_REMOVE:
            continue

        dest_lines.append('\n')
        dest_lines.extend(block)

    with open(dest, 'w') as f:
        f.writelines(dest_lines)


def convert_configs(torchtune_base_path: str,
                    lumino_job_configs_base_path: str):
    """Convert Torchtune Recipe Configs to Lumino equivalents.

    Given a Torchtune repository basepath and a Lumino job configs base path,
    convert various torchtune recipes into their Lumino equivalents
    (as per the whitelisted sub-dirs in `TORCHTUNE_RECIPE_MODELS`) 
    """
    recipe_configs_path = os.path.join(torchtune_base_path, TORCHTUNE_RECIPE_CONFIG_SUFFIX)
    whitelist = [f'{recipe_configs_path}/{m}' for m in TORCHTUNE_RECIPE_MODELS]
    for path, _, files in os.walk(recipe_configs_path):
        if path not in whitelist:
            continue
        for filename in files:
            src_filename = os.path.join(path, filename)
            dest_filename = os.path.join(lumino_job_configs_base_path,
                                         os.path.basename(path),
                                         filename.replace('yaml', 'yml'))
            print(f'src: {src_filename}, dest: {dest_filename}')
            _convert_config(src_filename, dest_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync torchtune recipe configs")
    parser.add_argument('-ttbp', '--torchtune_base_path', type=str, required=True,
                        help='Base path of `torchtune` repo')
    parser.add_argument('-dest', '--destination', type=str, required=False,
                        default='job-configs/torchtune/',
                        help='Base path of Lumino `torchtune` configs')
    args = parser.parse_args()
    convert_configs(args.torchtune_base_path, args.destination)
