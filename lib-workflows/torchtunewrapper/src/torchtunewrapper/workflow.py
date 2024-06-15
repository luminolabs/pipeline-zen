import importlib
from functools import partial
from logging import Logger
from typing import Optional, Callable

from omegaconf import DictConfig, OmegaConf
from torchtune.datasets._instruct import instruct_dataset

from common.model.factory import model_factory
from common.utils import load_job_config, get_or_generate_job_id, setup_logger, read_job_config_from_file, \
    get_logs_path, get_results_path, save_job_results


def run(job_config: DictConfig, tt_config: DictConfig, tt_recipe_fn: Callable, logger: Logger) -> dict:
    """
    Trains a model using torchtune recipies

    :param job_config: The job configuration
    :param tt_config: The torchtune specific job configuration
    :param tt_recipe_fn: The torchtune recipe function (the callable, not the name of the function)
    :param logger: The logger instance
    :return: The final loss value
    """
    job_id = job_config['job_id']

    # TODO: Possibly implement custom torchtune logger
    # TODO: Stream logs to cloud logging
    # TODO: Save the fine-tuned model weights to cloud storage

    # Fetch and load the base model
    m = model_factory(model_kind='llm', model_base=job_config['model_base'], logger=logger)
    # Update the base model path in the torchtune configuration
    tt_config = OmegaConf.merge(tt_config, {'base_model_path': m.name_or_path})  # it's the path in this case
    # Run the torchtune recipe, which will fine-tune the model
    loss = tt_recipe_fn(tt_config)

    # Save and return the results
    results = {'loss': loss}
    save_job_results(job_id, results, 'torchtune')
    logger.info('The job id was: ' + job_id)
    return results


def import_recipe_main(recipe: str) -> Callable:
    # Dynamically import the recipe module
    module = importlib.import_module(f'torchtunewrapper.recipes.{recipe}')
    # Access the function from the recipe module
    return getattr(module, 'recipe_main')


def main(job_config_name: str, job_id: Optional[str],
         dataset_id: str,
         batch_size: int = 1, shuffle: bool = True):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_name: The job configuration id; configuration files are found under `job-configs`
    :param job_id: The job id to use for logs, results, etc
    :param dataset_id: The dataset to use for training
    :param batch_size: Training batch size, default is 1
    :param shuffle: Whether to shuffle the dataset or not, default is True
    :return: The path to the fine-tuned model weights; which is the input to the evaluate workflow
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Overwrite job config values with values from input, if any
    job_config['job_id'] = job_id = get_or_generate_job_id(job_config_name, job_id)

    # Torchtune configuration
    tt_config = read_job_config_from_file(
        job_config['torchtune_config'],
        overrides={
            'logs_path': get_logs_path(job_id), 'output_dir': get_results_path(job_id),
            'shuffle': shuffle, 'batch_size': batch_size
        },
        is_torchtune=True)

    # Instantiate dataset
    dataset = instruct_dataset(
        tokenizer=None,
        source=dataset_id,
        template="torchtune.data.AlpacaInstructTemplate",
        train_on_input=job_config.get('train_on_input', False),
        max_seq_len=job_config.get('max_seq_len', None),
        packed=job_config.get('packed', False),
        split=job_config.get('split', 'train'),
    )

    # Get the torchtune recipe function
    tt_recipe_fn = import_recipe_main(job_config['torchtune_recipe'])
    tt_recipe_fn = partial(tt_recipe_fn, dataset=dataset)

    # Instantiate the main logger
    logger = setup_logger('torchtune_train_workflow', job_id)
    # Run the `torchtune` workflow, and handle unexpected exceptions
    try:
        return run(job_config, tt_config, tt_recipe_fn, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
