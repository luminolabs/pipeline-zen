import argparse
import os
from logging import Logger

import torch
import numpy as np
from torch import Tensor

from common.agents.model_scores import EvaluateScoresAgent
from common.scores.utils import mask_scores, scalar_scores
from common.utils import load_job_config, setup_logger, get_root_path, get_system_timestamp
from common.helpers import configure_model_and_dataloader
from common.tokenizer.utils import tokenize_inputs

# Point application to the `pipeline-zen_dev` GCP credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(get_root_path(), '.secrets', 'gcp_key.json')


def _evaluate(job_config: dict,  model_weights_id: str, logger: Logger):
    """
    Evaluates a model

    :param job_config: The job configuration
    :param model_weights_id: Which model weights to use for inference
    :param logger: The logging object
    :return:
    """
    # Grab and log the job id
    job_id = job_config['job_id']
    logger.info('The job id is: ' + job_id)

    # A logger for logging scores
    scores_logger = setup_logger('evaluate_workflow_metrics', job_id)

    # Setup logging and bigquery agent for scores
    scores_agent = EvaluateScoresAgent(job_id, scores_logger)

    # Log system specs
    scores_agent.log_system_specs()

    # Log job configuration
    scores_agent.log_job_config(job_config)

    model, dataloader, tokenizer, device = \
        configure_model_and_dataloader(job_config, logger,
                                       for_inference=True, model_weights=model_weights_id)

    # Log the start time
    scores_agent.mark_time_start()

    # Variables to store predictions and actual labels
    all_preds = []
    all_labels = []
    batch_cnt = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            model_args = {}
            if tokenizer:
                inputs = tokenize_inputs(inputs, tokenizer, model_args, device,
                                         **job_config.get('tokenizer_args', {}))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, **model_args)
            if isinstance(outputs, Tensor):
                predicted = outputs
            else:
                _, predicted = torch.max(outputs.logits, 1)
            # Store predictions and actual labels
            # TODO: Can we avoid converting to numpy?
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Log run information
            logger.info(f'Batch {batch_cnt + 1}/{len(dataloader)}')
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop the loop before the actual end of the dataset is reached
            if job_config.get('num_batches') == batch_cnt + 1:
                logger.info(f'Reached `num_batches` limit: {job_config.get("num_batches")}')
                break
            batch_cnt += 1

    # Convert lists to numpy arrays for scores calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate scores
    if len(all_preds.shape) == 1:
        accuracy, precision, recall, f1 = scalar_scores(all_preds, all_labels)
    else:
        accuracy, precision, recall, f1 = mask_scores(torch.tensor(all_preds), torch.tensor(all_labels))

    # Log scores
    scores_agent.log_scores(accuracy, precision, recall, f1,
                            stopped_at=batch_cnt + 1, num_batches=len(dataloader))

    # Log the end time
    scores_agent.mark_time_end()
    # Log the total training time
    scores_agent.log_time_elapsed()


def main(job_config_name: str, model_weights: str, job_id: str, batch_size: int, num_batches: int):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_name: The job configuration id; configuration files are found under `job_configs`
    :param model_weights: Which model weights to use for inference
    :param job_id: The job id to use for logs, results, etc.
    :param batch_size: The batch size to split the data into
    :param num_batches: How many batches to run on each epoch
    :return:
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Overwrite job config values with values from input, if any
    if job_id:
        job_config['job_id'] = job_id
    else:
        timestamp = get_system_timestamp()
        job_config['job_id'] = job_config['job_id'] + '-' + timestamp
    if batch_size:
        job_config['batch_size'] = batch_size
    if num_batches:
        job_config['num_batches'] = num_batches

    # Instantiate the main logger
    logger = setup_logger('evaluate_workflow', job_config["job_id"])
    # Run the `evaluate` workflow, and handle unexpected exceptions
    try:
        _evaluate(job_config, model_weights, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model evaluation workflow")
    parser.add_argument('-jc', '--job_config_name', nargs=1, type=str, required=True,
                        help='The name of the job config file, without the `.py` extension')
    parser.add_argument('-mw', '--model_weights', nargs=1, type=str, required=True,
                        help='The model weights file to use')
    parser.add_argument('-jid', '--job_id', nargs=1, type=str, required=False,
                        help='The job_id to use for evaluation; '
                             'logs and other job results and artifacts will be named after this.')
    parser.add_argument('-bs', '--batch_size', nargs=1, type=int, required=False,
                        help='The batch size to use for training ')
    parser.add_argument('-nb', '--num_batches', nargs=1, type=int, required=False,
                        help='The number of batches to run. This is helpful when testing code changes;'
                             'the trainer will stop after this many batches, and continue to the'
                             'next epoch')
    args = parser.parse_args()

    job_config_name = args.job_config_name and args.job_config_name[0]
    model_weights = args.model_weights and args.model_weights[0]
    job_id = args.job_id and args.job_id[0]
    batch_size = args.batch_size and args.batch_size[0]
    num_batches = args.num_batches and args.num_batches[0]

    return job_config_name, model_weights, job_id, batch_size, num_batches


# Run evaluate workflow
main(*parse_args())
