import asyncio
import sys

import torch
import numpy as np
from torch import Tensor

from common.metrics.utils import mask_metrics, scalar_metrics
from common.utils import load_job_config, setup_logger
from common.helpers import configure_model_and_dataloader
from common.tokenizer.utils import tokenize_inputs


async def main(job_config_id: str, model_weights_id: str):
    # Load job configuration
    job_config = load_job_config(job_config_id)
    job_id = job_config["job_id"]

    # A logger for logging metrics
    metrics = setup_logger('evaluate_metrics', job_id)
    # and a logger for logging everything else
    logger = setup_logger('evaluate_logger', job_id)

    model, dataloader, tokenizer, device = \
        configure_model_and_dataloader(job_config, logger,
                                       for_inference=True, model_weights_id=model_weights_id)

    # Variables to store predictions and actual labels
    all_preds = []
    all_labels = []
    batch_cnt = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_cnt += 1
            model_args = {}
            if tokenizer:
                inputs = tokenize_inputs(inputs, tokenizer, model_args, device,
                                         **job_config.get('tokenizer_args'))
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
            logger.info(f'Batch {batch_cnt}/{len(dataloader)}')
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop the loop before the actual end of the dataset is reached
            if job_config.get('num_batches') == batch_cnt:
                break

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    if len(all_preds.shape) == 1:
        accuracy, precision, recall, f1 = scalar_metrics(all_preds, all_labels)
    else:
        accuracy, precision, recall, f1 = mask_metrics(torch.tensor(all_preds), torch.tensor(all_labels))

    # Log metrics
    metrics.info(f'Accuracy: {accuracy}')
    metrics.info(f'Precision: {precision}')
    metrics.info(f'Recall: {recall}')
    metrics.info(f'F1 Score: {f1}')


job_config_id = sys.argv[1]
model_weights_id = sys.argv[2]
asyncio.run(main(job_config_id, model_weights_id))
