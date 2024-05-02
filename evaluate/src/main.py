import os
import sys
from logging import Logger

import torch
import numpy as np
from torch import Tensor

from common.agents.model_scores import EvaluateScoresAgent
from common.scores.utils import mask_scores, scalar_scores
from common.utils import load_job_config, setup_logger, get_root_path
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
    job_id = job_config["job_id"]

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
                                       for_inference=True, model_weights_id=model_weights_id)

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


def main(job_config_id: str, model_weights_id: str):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_id: The job configuration id; configuration files are found under `job_configs`
    :return:
    """
    # Load job configuration
    job_config = load_job_config(job_config_id)
    job_id = job_config["job_id"]
    # Instantiate the main logger
    logger = setup_logger('evaluate_workflow', job_id)
    # Run the `evaluate` workflow, and handle unexpected exceptions
    try:
        _evaluate(job_config, model_weights_id, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex


main(job_config_id=sys.argv[1], model_weights_id=sys.argv[2])
