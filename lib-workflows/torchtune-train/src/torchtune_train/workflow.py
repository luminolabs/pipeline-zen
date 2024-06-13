import importlib
import os
from logging import Logger
from typing import Optional, Callable

from omegaconf import OmegaConf, DictConfig

from common.model.utils import model_factory
from common.utils import load_job_config, get_or_generate_job_id, setup_logger, read_job_config_from_file


def run(job_config: DictConfig, tt_config: DictConfig, tt_recipe_fn: Callable, logger: Logger) -> dict:
    model_factory(model_kind=None, model_base=job_config['model_base'], logger=logger, token='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')
    r = tt_recipe_fn(tt_config)
    return {}


def import_recipe_main(recipe: str) -> Callable:
    # Dynamically import the recipe module
    module = importlib.import_module(f'torchtune_train.recipes.{recipe}')
    # Access the function from the recipe module
    return getattr(module, 'recipe_main')


def main(job_config_name: str, job_id: Optional[str]):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_name: The job configuration id; configuration files are found under `job-configs`
    :param job_id: The job id to use for logs, results, etc
    :return: The path to the fine-tuned model weights; which is the input to the evaluate workflow
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Overwrite job config values with values from input, if any
    job_config['job_id'] = job_id = get_or_generate_job_id(job_config_name, job_id)

    # Torchtune configuration
    tt_config = read_job_config_from_file(job_config['torchtune_config'], is_torchtune=True)
    tt_recipe_fn = import_recipe_main(job_config['torchtune_recipe'])

    # Instantiate the main logger
    logger = setup_logger('torchtune_train_workflow', job_id)
    # Run the `torchtune_train` workflow, and handle unexpected exceptions
    try:
        return run(job_config, tt_config, tt_recipe_fn, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
