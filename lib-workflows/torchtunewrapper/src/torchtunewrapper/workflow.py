from functools import partial
from logging import Logger
from typing import Optional

from celery.utils import worker_direct
from omegaconf import DictConfig, OmegaConf
from torch.distributed.launcher import elastic_launch, LaunchConfig
from torch.utils.data import Dataset
from torchtune.datasets import chat_dataset

from common.agent.model_scores import TorchtunewrapperMetricsAgent
from common.comms import heartbeat_wrapper
from common.dataset.base import dataset_provider_factory
from common.model.base import model_provider_factory
from common.utils import load_job_config, setup_logger, read_job_config_from_file, get_work_dir, save_job_results
from torchtunewrapper.recipes.mixtral_8x7b_fix import update_convert_weights_from_hf
from torchtunewrapper.utils import import_torchtune_recipe_fn, get_torchtune_config_filename

# Update the convert weights function to support the Mixtral-8x7B model
update_convert_weights_from_hf()


@heartbeat_wrapper('torchtunewrapper', 'download_dataset')
def _download_dataset(job_config: DictConfig, logger: Logger) -> Dataset:
    """
    Load the dataset for training;
    this will download the dataset from GCS if the dataset id is a GCS path otherwise
    it will pull the dataset from HuggingFace

    :param job_config: The job configuration
    :param logger: The logger instance
    :return:
    """
    chat_dataset_partial = partial(
        chat_dataset,
        tokenizer=None,
        conversation_style="openai",
        chat_format='torchtune.data.ChatMLFormat',
        split=job_config.get('split', 'train'),
        max_seq_len=job_config.get('max_seq_len', None),
        train_on_input=job_config.get('train_on_input', False),
        packed=job_config.get('packed', False))

    # Instantiate the dataset provider and download the dataset
    dataset_path = dataset_provider_factory(url=job_config['dataset_id'],
                                                job_id=job_config['job_id'], user_id=job_config['user_id'],
                                                logger=logger)()
    # Instantiate the chat dataset to use the downloaded dataset
    dataset = chat_dataset_partial(
        source="json",
        data_files=dataset_path
    )
    return dataset


@heartbeat_wrapper('torchtunewrapper', 'download_model')
def _download_model(model_base: str, logger: Logger) -> str:
    return model_provider_factory(url=model_base, logger=logger)()


def run(job_id: str, user_id: str, job_config: DictConfig, tt_config: DictConfig, logger: Logger) -> dict:
    """
    Trains a model using torchtune recipies

    :param job_id: The job id for the job
    :param user_id: The user id for the job
    :param job_config: The job configuration
    :param tt_config: The torchtune specific job configuration
    :param logger: The logger instance
    :return: The final loss value
    """
    # A logger for logging scores; also propagates to main logger
    scores_logger = setup_logger('torchtunewrapper_workflow.metrics', job_id, user_id, add_stdout=False)
    # Setup logging and bigquery agent for scores
    scores_agent = TorchtunewrapperMetricsAgent(job_id, user_id, scores_logger)

    # Log a few things about this job
    scores_logger.info('The job id is: ' + job_id)
    scores_agent.log_system_specs()
    scores_agent.log_job_config(dict(job_config))

    # Load the dataset
    dataset = _download_dataset(job_config, logger)

    # Check if we are using a single device
    is_single_device = job_config['num_gpus'] == 1

    # Fetch and load the base model
    model_path = _download_model(job_config['model_base'], logger)
    # Update the base model path in the torchtune configuration
    tt_config = OmegaConf.merge(tt_config, {'base_model_path': model_path})  # path, not name

    # Get the torchtune recipe function
    tt_recipe_fn_orig = import_torchtune_recipe_fn(job_config['use_lora'], is_single_device)
    tt_recipe_fn = partial(tt_recipe_fn_orig, job_id=job_id, user_id=user_id, cfg=tt_config, dataset=dataset)

    # Run the torchtune recipe, which will fine-tune the model
    if is_single_device:
        # Run the recipe on a single device
        tt_recipe_fn()
    else:
        # Run the recipe on multiple devices
        logger.info(f'Number of GPUs: {job_config["num_gpus"]}')
        # Set the name of the recipe function to the original function name;
        # because `partial` doesn't preserve the original function name
        tt_recipe_fn.__name__ = tt_recipe_fn.__qualname__ = tt_recipe_fn_orig.__name__
        # Instantiate the recipe on multiple devices
        tt_recipe_fn_multi = elastic_launch(
            config=LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=job_config['num_gpus'],
                rdzv_backend='c10d',
                rdzv_endpoint='localhost:0',
            ),
            entrypoint=tt_recipe_fn,
        )
        # Run the recipe
        tt_recipe_fn_multi()

    # Save and return the results
    results = {'see logs': 'see logs for results'}
    save_job_results(job_id, user_id, results, 'torchtunewrapper')
    logger.info('The job id was: ' + job_id)
    return results


def main(job_id: str, user_id: str, job_config_name: str,
         dataset_id: str = Optional[None],
         batch_size: int = 1, shuffle: bool = True, num_epochs: int = 1,
         use_lora: bool = True, use_qlora: bool = False,
         lr: float = 3e-4, seed: Optional[int] = None,
         num_gpus: int = 1,
         pytorch_cuda_alloc_conf: str = None):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_id: The job id to use for logs, results, etc.
    :param user_id: The user id for the job
    :param job_config_name: The job configuration id; configuration files are found under `job-configs`
    :param dataset_id: The dataset identifier, ex: `tatsu-lab/alpaca`
    :param batch_size: The training batch size, default is 1
    :param shuffle: Whether to shuffle the dataset or not, default is True
    :param num_epochs: Number of epochs to train
    :param use_lora: Whether to train with LoRA or do full training
    :param use_qlora: Whether to use QLoRA or not
    :param lr: The learning rate to use for training
    :param seed: The random seed to use for training
    :param num_gpus: The number of GPUs to use for training
    :param pytorch_cuda_alloc_conf: The PyTorch CUDA allocation configuration
    :return: The path to the fine-tuned model weights; which is the input to the evaluate workflow
    """

    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Set the job_id and user_id
    job_config['job_id'] = job_id
    job_config['user_id'] = user_id

    # Overwrite job config values with values from input, if any
    job_config.setdefault('dataset_id', dataset_id)
    job_config.setdefault('batch_size', batch_size)
    job_config.setdefault('shuffle', shuffle)
    job_config.setdefault('num_epochs', num_epochs)
    job_config.setdefault('use_lora', use_lora)
    job_config.setdefault('use_qlora', use_qlora)
    job_config.setdefault('lr', lr)
    job_config.setdefault('seed', seed)
    job_config.setdefault('num_gpus', num_gpus)
    job_config.setdefault('pytorch_cuda_alloc_conf', pytorch_cuda_alloc_conf)

    # Instantiate the main logger
    logger = setup_logger('torchtunewrapper_workflow', job_id, user_id)

    # Check if we are using a single device
    is_single_device = job_config['num_gpus'] == 1

    # If we are not using LoRA, we cannot use QLoRA either
    if not job_config['use_lora']:
        job_config['use_qlora'] = False

    # Set the work directory, to save logs, results, etc.
    work_dir = get_work_dir(job_id, user_id)

    # Load torchtune configuration
    tt_config_file = get_torchtune_config_filename(
        job_config['model_base'],
        job_config['use_lora'], job_config['use_qlora'],
        is_single_device)
    tt_config = read_job_config_from_file(
        tt_config_file,
        overrides={
            'logs_path': work_dir,
            'output_dir': work_dir,
            'epochs': job_config['num_epochs'],
            'shuffle': job_config['shuffle'],
            'batch_size': job_config['batch_size'],
            'pytorch_cuda_alloc_conf': job_config['pytorch_cuda_alloc_conf'],
            'lr': job_config['lr'],
            'seed': job_config['seed'],
        },
        sub_dir='torchtune')

    # Run the `torchtune` workflow, and handle unexpected exceptions
    try:
        return run(job_id, user_id, job_config, tt_config, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
