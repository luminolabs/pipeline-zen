import importlib
from typing import Callable


def import_torchtune_recipe_fn(use_lora: bool, use_single_device: bool) -> Callable:
    """
    :return: The imported torchtune recipe function
    """
    # Build recipe name
    finetune_type = 'lora' if use_lora else 'full'
    device_type = 'single_device' if use_single_device else 'distributed'
    recipe = f'{finetune_type}_finetune_{device_type}'
    # Dynamically import the recipe module
    module = importlib.import_module(f'torchtunewrapper.recipes.{recipe}')
    # Access the function from the recipe module
    return getattr(module, 'recipe_main')


def get_torchtune_config_filename(model_base: str, use_lora: bool, use_single_device: bool) -> str:
    """
    :return: The torchtune config filename
    """
    # Map model base to config prefix;
    # also serves as a check for supported bases
    model_base_to_config_prefix = {
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama3/8B',
        'meta-llama/Meta-Llama-3-70B-Instruct': 'llama3/70B',
        'mistralai/Mistral-7B-Instruct-v0.1': 'mistral/7B',
        'mistralai/Mistral-7B-Instruct-v0.3': 'mistral/7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mistral/8x7B',  # yes, `Mixtral` is intentional
        'meta-llama/Meta-Llama-3-8B': 'llama3/8B',
        'meta-llama/Meta-Llama-3-70B': 'llama3/70B',
        'mistralai/Mistral-7B-v0.1': 'mistral/7B',
        'mistralai/Mistral-7B-v0.3': 'mistral/7B',
    }
    # Raise error if model base is not supported
    if model_base not in model_base_to_config_prefix:
        raise ValueError(f'Unsupported model base: {model_base}; supported bases are: '
                         f'{", ".join(model_base_to_config_prefix.keys())}')
    # Return the config filename
    return (f'{model_base_to_config_prefix[model_base]}_'
            f'{"lora" if use_lora else "full"}'
            f'{"_single_device" if use_single_device else ""}.yml')
