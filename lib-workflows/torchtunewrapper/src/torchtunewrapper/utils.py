import importlib
from logging import Logger
from typing import Callable, Type

import requests
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.mistral import MistralTokenizer

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.config_manager import config
from common.gcp import send_heartbeat as _send_heartbeat
from common.utils import setup_logger
from torchtunewrapper.recipes.recipe_base import RecipeBase

# Supported tokenizers - this is a union of all tokenizers used in the recipes that we support
Tokenizers = Llama3Tokenizer | MistralTokenizer


def send_heartbeat(job_id: str, user_id: str, task: str, status: str) -> None:
    """
    Send a heartbeat message to the pipeline-zen-jobs-heartbeats topic.

    :param job_id: The job id
    :param user_id: The user id
    :param task: The task name
    :param status: The status of the job
    """
    status = f"wf-torchtune-{task}-{status}"
    _send_heartbeat(job_id, user_id, status)
    pass


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


def _count_dataset_tokens(dataset: Dataset) -> int:
    """
    Count the number of tokens in the dataset.
    """
    return sum([len(record['tokens']) for record in dataset])


def _log_tokens_and_check_user_credits(job_id: str, user_id: str,
                                       token_count: int, num_epochs: int, logger: Logger) -> bool:
    """
    Log the number of tokens in the dataset to the API, and check if the user has enough credits to run the job.
    """
    # If we're mocking user credits, or calls to Customer API are disabled, return True
    # This is useful for local testing
    if config.mock_user_has_enough_credits or not config.customer_api_enabled:
        return True

    api_url = f"{config.customer_api_url}/billing/credits-deduct"
    headers = {
        "x-api-key": f"{config.customer_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "user_id": user_id,
        "fine_tuning_job_id": job_id,
        "usage_amount": token_count * num_epochs,
        "usage_unit": "TOKEN",
        "service_name": "FINE_TUNING_JOB",
    }
    logger.info(f"Logging token count to API: {payload}")
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("has_enough_credits", False)


def run_recipe(recipe_class: Type[RecipeBase], job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset) -> None:
    """
    Run a torchtune recipe.
    """
    # Set up the main logger
    logger = setup_logger("torchtunewrapper_recipe", job_id, user_id)
    # A logger for logging scores; also propagates to main logger
    scores_logger = setup_logger('torchtunewrapper_recipe.metrics', job_id, user_id, add_stdout=False)
    # Setup logging and bigquery agent for scores
    scores_agent = TorchtunewrapperScoresAgent(job_id, scores_logger)
    # Initialize the recipe and start training
    recipe = recipe_class(job_id, user_id, cfg, dataset, logger, scores_agent)
    recipe.setup()
    # Count tokens and check if user has enough credits to run the job
    # Note: We can only do this after the recipe is set up, because the tokenizer needs to be initialized first
    token_count = _count_dataset_tokens(recipe.dataset)
    has_enough_credits = _log_tokens_and_check_user_credits(
        job_id, user_id, token_count, recipe.total_epochs, logger)
    if not has_enough_credits:
        raise PermissionError(f"User does not have enough credits to run the job; "
                              f"job_id: {job_id}, user_id: {user_id}, token_count: {token_count}")
    # Begin training
    recipe.train()
    # Save the checkpoint and cleanup
    recipe.save_checkpoint()
    recipe.cleanup()


def get_torchtune_config_filename(model_base: str,
                                  use_lora: bool, use_qlora: bool,
                                  use_single_device: bool) -> str:
    """
    :return: The torchtune config filename
    """
    # Map model base to config prefix;
    # also serves as a check for supported bases
    model_base_to_config_prefix = {
        # Llama 3.1 Instruct
        'meta-llama/Meta-Llama-3.1-8B-Instruct': 'llama3_1/8B',
        'meta-llama/Meta-Llama-3.1-70B-Instruct': 'llama3_1/70B',
        # Llama 3.1 base
        'meta-llama/Meta-Llama-3.1-8B': 'llama3_1/8B',
        'meta-llama/Meta-Llama-3.1-70B': 'llama3_1/70B',
        # Mistral v0.1 Instruct
        'mistralai/Mistral-7B-Instruct-v0.1': 'mistral/7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mistral/8x7B',  # yes, `Mixtral` is intentional
        # Mistral v0.1 base
        'mistralai/Mistral-7B-v0.1': 'mistral/7B',
        'mistralai/Mistral-7B-Instruct-v0.3': 'mistral/7B',
        # Mistral v0.3 base
        'mistralai/Mistral-7B-v0.3': 'mistral/7B',
    }
    # Raise error if model base is not supported
    if model_base not in model_base_to_config_prefix:
        raise ValueError(f'Unsupported model base: {model_base}; supported bases are: '
                         f'{", ".join(model_base_to_config_prefix.keys())}')
    # Return the config filename
    return (f'{model_base_to_config_prefix[model_base]}_'
            f'{"q" if use_qlora else ""}'
            f'{"lora" if use_lora else "full"}'
            f'{"_single_device" if use_single_device else ""}.yml')
