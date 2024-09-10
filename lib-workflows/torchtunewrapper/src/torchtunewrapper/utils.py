import importlib
from typing import Callable

from omegaconf import DictConfig
from torch.utils.data import Dataset

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.gcp import send_heartbeat as _send_heartbeat
from common.utils import setup_logger


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


def run_recipe(recipe_class, job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset) -> None:
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
    recipe = recipe_class(cfg, logger, scores_agent, dataset, is_lora=True)
    recipe.setup()
    recipe.train()
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
