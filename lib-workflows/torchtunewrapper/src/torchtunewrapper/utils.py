import importlib
from logging import Logger
from typing import Callable, Type

import requests
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchtune import config as tt_config

from common.agent.job_logger import TorchtunewrapperLoggerAgent
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
    # user_id=0x... is used for protocol jobs, which are paid differently
    if user_id in ("0", "-1") or user_id.startswith("0x"):
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
    # Setup job logger agent for streaming to multiple channels
    job_logger_agent = TorchtunewrapperLoggerAgent(
        job_id, user_id,
        agent_logger=setup_logger('torchtunewrapper_logger', job_id, user_id),
        main_logger=logger)

    # Instantiate the tokenizer and set it in the dataset
    tokenizer = tt_config.instantiate(cfg.tokenizer)
    dataset._model_transform = tokenizer
    # Count tokens and check if user has enough credits to run the job
    token_count = _count_dataset_tokens(dataset)
    has_enough_credits = _deduct_api_user_credits(job_id, user_id, token_count,
                                                  cfg.epochs, logger)
    if not has_enough_credits:
        raise PermissionError(f"User does not have enough credits to run the job; "
                              f"job_id: {job_id}, user_id: {user_id}, token_count: {token_count}")

    # Initialize and set up the recipe, train the model, and save the checkpoint
    recipe = recipe_class(job_id, user_id, cfg, dataset, logger, job_logger_agent)
    recipe.setup()
    recipe.train()
    recipe.save_checkpoint(cfg.epochs)
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
