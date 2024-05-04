import argparse
import os

from common.utils import load_job_config, setup_logger, get_root_path, get_system_timestamp
from train import run

# Point application to the `pipeline-zen_dev` GCP credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(get_root_path(), '.secrets', 'gcp_key.json')


def main(job_config_name: str, job_id: str, batch_size: int, num_epochs: int, num_batches: int):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_name: The job configuration id; configuration files are found under `job_configs`
    :param job_id: The job id to use for logs, results, etc
    :param batch_size: The batch size to split the data into
    :param num_epochs: The number of epochs to train on
    :param num_batches: How many batches to run on each epoch
    :return:
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Overwrite job config values with values from input, if any
    if job_id:
        job_config['job_id'] = job_id
    else:
        timestamp =get_system_timestamp()
        job_config['job_id'] = job_config['job_id'] + '-' + timestamp
    if batch_size:
        job_config['batch_size'] = batch_size
    if num_epochs:
        job_config['num_epochs'] = num_epochs
    if num_batches:
        job_config['num_batches'] = num_batches

    # Instantiate the main logger
    logger = setup_logger('train_workflow', job_config['job_id'])
    # Run the `train` workflow, and handle unexpected exceptions
    try:
        run(job_config, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model training workflow")
    parser.add_argument('-jc', '--job_config_name', nargs=1, type=str, required=True,
                        help="The name of the job config file, without the `.py` extension")
    parser.add_argument('-jid', '--job_id', nargs=1, type=str, required=False,
                        help="The job_id to use for training; "
                             "logs and other job results and artifacts will be named after this.")
    parser.add_argument('-bs', '--batch_size', nargs=1, type=int, required=False,
                        help="The batch size to use for training ")
    parser.add_argument('-ne', '--num_epochs', nargs=1, type=int, required=False,
                        help="The number of epochs to train the model")
    parser.add_argument('-nb', '--num_batches', nargs=1, type=int, required=False,
                        help="The number of batches to run. This is helpful when testing code changes;"
                             "the trainer will stop after this many batches, and continue to the"
                             "next epoch")
    args = parser.parse_args()

    job_config_name = args.job_config_name and args.job_config_name[0]
    job_id = args.job_id and args.job_id[0]
    batch_size = args.batch_size and args.batch_size[0]
    num_epochs = args.num_epochs and args.num_epochs[0]
    num_batches = args.num_batches and args.num_batches[0]

    return job_config_name, job_id, batch_size, num_epochs, num_batches


# Run train workflow
main(*parse_args())
