import importlib
from logging import Logger
from typing import Callable, Type

import requests
from omegaconf import DictConfig
from torch.utils.data import Dataset

from common.agent.model_scores import TorchtunewrapperMetricsAgent
from common.config_manager import config
from common.utils import setup_logger
from torchtunewrapper.recipes.recipe_base import RecipeBase


def import_torchtune_recipe_fn(use_lora: bool, use_single_device: bool, job_config_id: str) -> Callable:
    """
    Magic function to import the torchtune recipe function dynamically.
    """
    # If job_config_id is 'llm_dummy', return the Dummy recipe
    if job_config_id == 'llm_dummy':
        module = importlib.import_module('torchtunewrapper.recipes.dummy')
        return getattr(module, 'recipe_main')

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
    Count the number of tokens in a dataset.
    """
    return sum([len(record['tokens']) for record in dataset])


def _deduct_api_user_credits(job_id: str, user_id: str,
                             token_count: int, num_epochs: int, logger: Logger) -> bool:
    """
    Log the number of tokens in the dataset to the API, and check if the user has enough credits to run the job.
    """
    # If we're mocking user credits, or calls to Customer API are disabled, return True
    # This is useful for local testing
    if config.mock_user_has_enough_credits or not config.customer_api_enabled:
        logger.info("Skipping credit check due to config settings")
        return True

    # If user_id is 0 or -1, skip the credit check;
    # user_id=0|-1 is used for jobs that didn't originate from the
    # customer API, and were created internally
    if user_id in ("0", "-1"):
        logger.info(f"Skipping credit check for user_id={user_id}")
        return True

    api_url = f"{config.customer_api_url}{config.customer_api_credits_deduct_endpoint}"
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
    if response.status_code != 200:
        return False
    return True


def run_recipe(recipe_class: Type[RecipeBase], job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset) -> None:
    """
    Run a torchtune recipe.
    """
    # Set up the main logger
    logger = setup_logger("torchtunewrapper_recipe", job_id, user_id)
    # A logger for logging metrics; also propagates to main logger
    metrics_logger = setup_logger('torchtunewrapper_recipe.metrics', job_id, user_id, add_stdout=False)
    # Setup logging and bigquery agent for scores
    metrics_agent = TorchtunewrapperMetricsAgent(job_id, user_id, metrics_logger)
    # Initialize and set up the recipe
    recipe = recipe_class(job_id, user_id, cfg, dataset, logger, metrics_agent)
    # Set up the tokenizer now so that we can count the tokens in the dataset
    recipe.setup_tokenizer()
    # Count tokens and check if user has enough credits to run the job
    token_count = _count_dataset_tokens(recipe.dataset)
    has_enough_credits = _deduct_api_user_credits(job_id, user_id, token_count, recipe.total_epochs, logger)
    if not has_enough_credits:
        raise PermissionError(f"User does not have enough credits to run the job; "
                              f"job_id: {job_id}, user_id: {user_id}, token_count: {token_count}")
    recipe.setup()
    # Begin training
    recipe.train()
    # Save weights
    recipe.save_checkpoint()
    # Destroy multi-GPU and multi-node process groups
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
        'hf://meta-llama/Meta-Llama-3.1-8B-Instruct': 'llama3_1/8B',
        'hf://meta-llama/Meta-Llama-3.1-70B-Instruct': 'llama3_1/70B',
        # Mistral v0.1 Instruct
        'hf://mistralai/Mistral-7B-Instruct-v0.1': 'mistral/7B',
        # Dummy
        'hf://crumb/nano-mistral': 'dummy/nano_mistral',
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
