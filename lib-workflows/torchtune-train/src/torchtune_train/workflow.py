import importlib
from logging import Logger
from typing import Optional, Callable

from omegaconf import DictConfig, OmegaConf

from common.model.factory import model_factory
from common.utils import load_job_config, get_or_generate_job_id, setup_logger, read_job_config_from_file, \
    get_logs_path, get_results_path


def run(job_config: DictConfig, tt_config: DictConfig, tt_recipe_fn: Callable, logger: Logger) -> dict:
    # Fetch and load the base model
    m = model_factory(model_kind='llm', model_base=job_config['model_base'], logger=logger)
    # Update the base model path in the torchtune configuration
    tt_config = OmegaConf.merge(tt_config, {'base_model_path': m.name_or_path})  # it's the path in this case
    # Run the torchtune recipe, which will fine-tune the model
    loss = tt_recipe_fn(tt_config)
    # Return the loss value
    return {'loss': loss}


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
    tt_config = read_job_config_from_file(
        job_config['torchtune_config'],
        overrides={'logs_path': get_logs_path(job_id), 'results_path': get_results_path(job_id)},
        is_torchtune=True)
    tt_recipe_fn = import_recipe_main(job_config['torchtune_recipe'])

    # Instantiate the main logger
    logger = setup_logger('torchtune_train_workflow', job_id)
    # Run the `torchtune_train` workflow, and handle unexpected exceptions
    try:
        return run(job_config, tt_config, tt_recipe_fn, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
