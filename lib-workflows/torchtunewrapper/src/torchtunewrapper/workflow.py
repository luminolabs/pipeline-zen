from functools import partial
from logging import Logger
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from torch.distributed.launcher import elastic_launch, LaunchConfig
from torchtune.datasets._instruct import instruct_dataset

from common.dataset.provider.huggingface import HuggingFace
from common.model.factory import model_factory
from common.utils import load_job_config, get_or_generate_job_id, setup_logger, read_job_config_from_file, \
    get_logs_path, get_results_path, save_job_results
from torchtunewrapper.utils import import_torchtune_recipe_fn, get_torchtune_config_filename, \
    get_torchtune_dataset_template


def run(job_config: DictConfig, tt_config: DictConfig, logger: Logger) -> dict:
    """
    Trains a model using torchtune recipies

    :param job_config: The job configuration
    :param tt_config: The torchtune specific job configuration
    :param logger: The logger instance
    :return: The final loss value
    """
    job_id = job_config['job_id']

    # TODO: Possibly implement custom torchtune logger
    # TODO: Stream logs to cloud logging / bigquery
    # TODO: Unit tests
    # TODO: Put together detailed how to use instructions

    # Instantiate dataset
    dataset = instruct_dataset(
        tokenizer=None,
        source=job_config['dataset_id'],
        template=get_torchtune_dataset_template(job_config['dataset_template']),
        train_on_input=job_config.get('train_on_input', False),
        max_seq_len=job_config.get('max_seq_len', None),
        packed=job_config.get('packed', False),
        split=job_config.get('split', 'train'),
        cache_dir=HuggingFace.get_dataset_cache_dir(),
    )

    # Fetch and load the base model
    m = model_factory(model_kind='llm', model_base=job_config['model_base'], logger=logger)
    # Update the base model path in the torchtune configuration
    tt_config = OmegaConf.merge(tt_config, {'base_model_path': m.name_or_path})  # path, not name

    # Get the torchtune recipe function
    tt_recipe_fn_orig = import_torchtune_recipe_fn(job_config['use_lora'], job_config['use_single_device'])
    tt_recipe_fn = partial(tt_recipe_fn_orig, cfg=tt_config, dataset=dataset)

    # Run the torchtune recipe, which will fine-tune the model
    if job_config['use_single_device']:
        # Run the recipe on a single device
        tt_recipe_fn()
    else:
        # Run the recipe on multiple devices
        e = elastic_launch(
            config=LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=1,
                rdzv_backend='c10d',
                rdzv_endpoint='localhost:0',
            ),
            entrypoint=tt_recipe_fn,
        )
        tt_recipe_fn.__name__ = tt_recipe_fn.__qualname__ = tt_recipe_fn_orig.__name__
        e()

    # Save and return the results
    results = {'see logs': 'see logs for results'}
    save_job_results(job_id, results, 'torchtune')
    logger.info('The job id was: ' + job_id)
    return results


def main(job_config_name: str, job_id: Optional[str] = None,
         dataset_id: str = Optional[None], dataset_template: Optional[str] = None,
         batch_size: int = 1, shuffle: bool = True,
         num_epochs: int = 1, use_lora: bool = True, use_single_device: bool = True):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_name: The job configuration id; configuration files are found under `job-configs`
    :param job_id: The job id to use for logs, results, etc.
    :param dataset_id: The dataset identifier, ex:
    :param dataset_template: The dataset template to use for training; e.g. `instruct`, or `summarization`
    :param batch_size: The training batch size, default is 1
    :param shuffle: Whether to shuffle the dataset or not, default is True
    :param num_epochs: Number of epochs to train
    :param use_lora: Whether to train with LoRA or do full training
    :param use_single_device: Whether to train on a single or multiple GPU devices
    :return: The path to the fine-tuned model weights; which is the input to the evaluate workflow
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Overwrite job config values with values from input, if any
    job_config['job_id'] = job_id = get_or_generate_job_id(job_config_name, job_id)
    job_config.setdefault('dataset_id', dataset_id)
    job_config.setdefault('dataset_template', dataset_template)
    job_config.setdefault('batch_size', batch_size)
    job_config.setdefault('shuffle', shuffle)
    job_config.setdefault('num_epochs', num_epochs)
    job_config.setdefault('use_lora', use_lora)
    job_config.setdefault('use_single_device', use_single_device)

    # Load torchtune configuration
    tt_config_file = get_torchtune_config_filename(
        job_config['model_base'], job_config['use_lora'], job_config['use_single_device'])
    tt_config = read_job_config_from_file(
        tt_config_file,
        overrides={
            'logs_path': get_logs_path(job_config['job_id']),
            'output_dir': get_results_path(job_config['job_id']),
            'epochs': job_config['num_epochs'],
            'shuffle': job_config['shuffle'],
            'batch_size': job_config['batch_size']
        },
        is_torchtune=True)

    # Instantiate the main logger
    logger = setup_logger('torchtune_train_workflow', job_id)
    # Run the `torchtune` workflow, and handle unexpected exceptions
    try:
        return run(job_config, tt_config, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
