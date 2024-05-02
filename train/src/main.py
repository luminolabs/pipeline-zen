import argparse
import os
from logging import Logger

import torch
from torch import nn, optim, Tensor

from common.loss.utils import loss_factory
from common.utils import get_model_weights_path, load_job_config, setup_logger, get_root_path, \
    get_system_timestamp
from common.helpers import configure_model_and_dataloader
from common.tokenizer.utils import tokenize_inputs
from common.agents.model_scores import TrainScoresAgent

# Point application to the `pipeline-zen_dev` GCP credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(get_root_path(), '.secrets', 'gcp_key.json')


def _train(job_config: dict, job_config_id: str, logger: Logger):
    """
    Trains a model

    :param job_config: The job configuration
    :type job_config_id: The job config id that was entered on the command line
    :param logger: The logging object
    :return:
    """
    # Grab and log the job id
    job_id = job_config['job_id']
    logger.info('The job id is: ' + job_id)

    # A logger for logging scores
    scores_logger = setup_logger('train_workflow_metrics', job_id)

    # Setup logging and bigquery agent for scores
    scores_agent = TrainScoresAgent(job_id, scores_logger)

    # Log system specs
    scores_agent.log_system_specs()

    # Log job configuration
    scores_agent.log_job_config(job_config)

    model, dataloader, tokenizer, device = \
        configure_model_and_dataloader(job_config, logger)

    # Loss calculator
    criterion = loss_factory(
        job_config.get('loss_func_name'), logger,
        **job_config.get('loss_func_args', {}))
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=job_config.get('learning_rate'))
    logger.info("Loss and Optimizer is set")

    # Log the start time
    scores_agent.mark_time_start()

    for epoch_num in range(job_config.get('num_epochs')):
        running_loss = 0.0
        batch_num = 0
        model.train()
        # The dataloader will load a batch of records from the dataset
        for inputs, labels in dataloader:
            model_args = {}
            if tokenizer:
                inputs = tokenize_inputs(inputs, tokenizer, model_args, device)
            # Load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Run training logic
            optimizer.zero_grad()
            outputs = model(inputs, **model_args)
            if isinstance(outputs, Tensor):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Log training information
            scores_agent.log_batch(batch_num + 1, len(dataloader), loss.item(), epoch_num + 1,
                                   job_config.get("num_epochs"))
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop training loop before the actual end of the dataset is reached
            if job_config.get('num_batches') == batch_num + 1:
                break
            batch_num += 1
        # Log training information
        scores_agent.log_epoch(epoch_num + 1, job_config.get("num_epochs"), running_loss / len(dataloader))

    # Log the end time
    scores_agent.mark_time_end()
    # Log the total training time
    scores_agent.log_time_elapsed()

    logger.info("Training loop complete, now saving the model")
    # Save the trained model
    model_weights_path = get_model_weights_path(job_config.get('job_id'))
    torch.save(model.module.state_dict()
               if isinstance(model, nn.DataParallel)
               else model.state_dict(), model_weights_path)
    logger.info("Trained model saved! at: " + model_weights_path)


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
        _train(job_config, job_config_name, logger)
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
