from typing import List, Any
import os
import argparse
import random
from utils import load_json_files, partition_k, write_json_objects

SAMPLE_PERCENT_MIN = 20
SAMPLE_PERCENT_MAX = 100

HOLDOUT_PERCENT_MIN = 10
HOLDOUT_PERCENT_DEFAULT = 20
HOLDOUT_PERCENT_MAX = 30

NUM_FOLDS_MIN = 1
NUM_FOLDS_DEFAULT = 4
NUM_FOLDS_MAX = 5


def write_data_by_index(data: List[Any], indices: List[int], output_path: str):
    """Write selected items from a list to a JSON file based on the given indices.

    Args:
        data: List of items to select from
        indices: List of integer indices indicating which items to write
        output_path: File path where the JSON output should be written
    """
    idx_data = [data[idx] for idx in indices]
    print(f'Writing {len(idx_data)} entries to file: {output_path}')
    write_json_objects(idx_data, output_path)


def dataset_prep(dataset_dir: str,
                 seed: int,
                 sample_pct: int,
                 holdout_pct: int,
                 num_folds: int):
    # Set random seed
    random.seed(args.seed)
    # Can sample at-most the pct of data not going into the holdout set.
    sample_pct = min(sample_pct, 100 - holdout_pct)
    # Create output base path - all other data will be under here or in sub-directories.
    output_base_path = os.path.join(dataset_dir,
        f'dataset_seed_{seed}_sample_{sample_pct}_holdout_{holdout_pct}_k_{num_folds}')

    # Load the original data
    data_list = load_json_files(dataset_dir)
    num_data = len(data_list)
    print(f'Loaded data from {dataset_dir}: {num_data} entries')

    # Create a list of indices representing the data and shuffle.
    data_indices = list(range(num_data))
    random.shuffle(data_indices)

    # Create a list of indices for each dataset type: holdout and sample.
    holdout_size = int(num_data * (holdout_pct / 100))
    sample_size = int(num_data * (sample_pct / 100))

    # Create holdout set by random sampling and save.
    holdout_indices = random.sample(data_indices, holdout_size)
    write_data_by_index(data_list, holdout_indices, os.path.join(output_base_path, 'holdout.json'))

    # Remove holdout indices
    remaining_indices = list(set(data_indices) - set(holdout_indices))
    # Randomly sample from the remaining
    sample_indices = random.sample(remaining_indices, min(sample_size, len(remaining_indices)))
    # Now we partition the samples into the number of folds.
    sample_indices_partitioned = partition_k(sample_indices, num_folds)
    for i in range(num_folds):
       fold_path = os.path.join(output_base_path, f'fold_{i}')
       # For each fold, we compute the indices representing the test and training data.
       test_indices = sample_indices_partitioned[i]
       train_indices = list(set(sample_indices) - set(test_indices))
       write_data_by_index(data_list, train_indices, os.path.join(fold_path, 'train.json'))
       write_data_by_index(data_list, test_indices, os.path.join(fold_path, 'test.json'))


# Example command-line:  See README
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from a given dataset')
    parser.add_argument('-dsd', '--dataset_dir', type=str, required=True,
                        help='Dir containing data in JSON format')
    parser.add_argument('-s', '--seed', type=int, required=True, help='RNG seed to use')
    parser.add_argument('-sp', '--sample_percent', type=int, required=True,
                        help='Percentage of the data to sample (will be capped to: 100 - `holdout_percent`)')
    parser.add_argument('-hp', '--holdout_percent', type=int,
                        required=False, default=HOLDOUT_PERCENT_DEFAULT,
                        help='Percentage of the data to use for holdout set')
    parser.add_argument('-k', '--num_folds', type=int,
                        required=False, default=NUM_FOLDS_DEFAULT,
                        help='Number of K-folds for cross validation')
    args = parser.parse_args()

    if args.num_folds < NUM_FOLDS_MIN or args.num_folds > NUM_FOLDS_MAX:
       parser.error(f'`num_folds` must be in range [{NUM_FOLDS_MIN}, {NUM_FOLDS_MAX}]')

    if args.holdout_percent < HOLDOUT_PERCENT_MIN or args.holdout_percent > HOLDOUT_PERCENT_MAX:
       parser.error(f'`holdout_percent` must be in range [{HOLDOUT_PERCENT_MIN}, {HOLDOUT_PERCENT_MAX}]')

    if args.sample_percent < SAMPLE_PERCENT_MIN or args.sample_percent > SAMPLE_PERCENT_MAX:
       parser.error(f'`sample_percent` must be in range [{SAMPLE_PERCENT_MIN}, {SAMPLE_PERCENT_MAX}]')

    dataset_prep(args.dataset_dir,
                 args.seed,
                 args.sample_percent,
                 args.holdout_percent,
                 args.num_folds)
