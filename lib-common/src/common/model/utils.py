import os

from common.config_manager import config


def get_model_cache_dir(model_provider: str, model_name: str):
    """
    Where to store the base model locally
    ex: `.cache/models/huggingface/llama3-8b`

    :param model_provider: The model provider
    :param model_name: The model name
    :return: The path to the base model cache directory
    """
    return os.path.join(config.root_path, config.cache_path, 'models', model_provider, model_name.lower())


def add_hf_params(model_base: str, **kwargs) -> dict:
    """
    Add Hugging Face parameters to the model configuration.

    :param model_base: The base model name
    :param kwargs: Keyword arguments to pass to the model
    :return: Updated keyword arguments
    """
    if 'token' not in kwargs:
        kwargs['token'] = config.hf_token
    if 'local_dir' not in kwargs:
        kwargs['local_dir'] = get_model_cache_dir('huggingface', model_base)
    return kwargs
