import asyncio
import logging
import os
import sys
from datetime import datetime

import torch
from torch import nn, optim, Tensor

from common.loss.utils import loss_factory
from common.utils import get_model_weights_path, load_job_config, setup_logger
from common.helpers import configure_model_and_dataloader
from common.tokenizer.utils import tokenize_inputs


def _train(job_config: dict, job_id: str, job_config_id: str, logger: logging.Logger):
    """
    Trains a model

    :param job_config: The job configuration
    :param job_id: The job id
    :type job_config_id: The job config id that was entered on the command line
    :param logger: The logging object
    :return:
    """
    # A logger for logging metrics
    metrics = setup_logger('train_metrics', job_id)

    model, dataloader, tokenizer, device = \
        configure_model_and_dataloader(job_config, logger)

    # Loss calculator
    criterion = loss_factory(
        job_config.get('loss_func_name'), logger,
        **job_config.get('loss_func_args', {}))
    # Optimizer
    # TODO: Allow using different optimizers through configuration
    optimizer = optim.Adam(model.parameters(), lr=job_config.get('learning_rate'))
    logger.info("Loss and Optimizer is set")

    # Capture the start time
    start_time = datetime.now()
    metrics.info(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch_cnt in range(job_config.get('num_epochs')):
        running_loss = 0.0
        batch_cnt = 0
        model.train()
        # The dataloader will load a batch of records from the dataset
        for inputs, labels in dataloader:
            batch_cnt += 1
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
            metrics.info(f'Batch {batch_cnt}/{len(dataloader)}, Batch Loss: {loss.item()}')
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop training loop before the actual end of the dataset is reached
            if job_config.get('num_batches') == batch_cnt:
                break
        # Log training information
        metrics.info(f'Epoch {epoch_cnt + 1}/{job_config.get("num_epochs")}, Loss: {running_loss / len(dataloader)}')

    # Capture the end time
    end_time = datetime.now()
    metrics.info(f"Training ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate and print the total training time
    total_time = end_time - start_time
    total_minutes = total_time.total_seconds() / 60
    metrics.info(f"Total training time: {total_minutes:.2f} minutes")

    logger.info("Training loop complete, now saving the model")
    # Save the trained model
    model_weights_path = get_model_weights_path(job_config.get('job_id'))
    torch.save(model.module.state_dict()
               if isinstance(model, nn.DataParallel)
               else model.state_dict(), model_weights_path)
    logger.info("Trained model saved! at: " + model_weights_path)
    logger.info("... use these arguments to evaluate your model: `" +
                job_config_id + " " +
                os.path.basename(model_weights_path) + "`")


def main(job_config_id: str):
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_config_id: The job configuration id; configuration files are found under `job_configs`
    :return:
    """
    # Load job configuration
    job_config = load_job_config(job_config_id)
    job_id = job_config["job_id"]
    # Instantiate the main logger
    logger = setup_logger('train_logger', job_id)
    # Run the `train` workflow, and handle unexpected exceptions
    try:
        _train(job_config, job_id, job_config_id, logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex


main(job_config_id=sys.argv[1])
