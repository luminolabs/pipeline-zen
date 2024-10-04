from logging import Logger
from typing import Optional

import torch
from torch import optim, Tensor, nn

from common.agents.model_scores import TrainScoresAgent
from common.helpers import configure_model_and_dataloader
from common.loss.utils import loss_factory
from common.tokenizer.utils import tokenize_inputs
from common.utils import setup_logger, get_model_weights_path, load_job_config, get_or_generate_job_id, \
    save_job_results


def run(job_config: dict, logger: Logger) -> dict:
    """
    Trains a model

    :param job_config: The job configuration
    :param logger: The logger
    :return: The final epoch's loss value
    """
    job_id = job_config['job_id']
    user_id = job_config['user_id']

    # A logger for logging scores; also propagates to main logger
    scores_logger = setup_logger('train_workflow.metrics', job_id, user_id, add_stdout=False)

    # Setup logging and bigquery agent for scores
    scores_agent = TrainScoresAgent(job_id, user_id, scores_logger)

    # Log a few things about this job
    scores_logger.info('The job id is: ' + job_id)
    scores_agent.log_system_specs()
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

    epoch_loss = None
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
        epoch_loss = running_loss / len(dataloader)
        scores_agent.log_epoch(epoch_num + 1, job_config.get("num_epochs"), running_loss / len(dataloader))

    # Log the end time
    scores_agent.mark_time_end()
    # Log the total training time
    scores_agent.log_time_elapsed()

    # Save the trained model
    logger.info("Training loop complete, now saving the model")
    model_weights_path = get_model_weights_path(job_id, user_id)
    torch.save(model.module.state_dict()
               if isinstance(model, nn.DataParallel)
               else model.state_dict(), model_weights_path)
    logger.info("Trained model saved! at: " + model_weights_path)

    results = {'loss': epoch_loss}
    save_job_results(job_id, user_id, results, 'train')
    scores_logger.info('The job id was: ' + job_id)
    return results


def main(job_id: str, user_id: str, job_config_name: str,
         batch_size: Optional[int], num_epochs: Optional[int], num_batches: Optional[int]) -> dict:
    """
    Workflow entry point, mainly for catching unhandled exceptions

    :param job_id: The job id to use for logs, results, etc
    :param user_id: The user id for the job
    :param job_config_name: The job configuration id; configuration files are found under `job-configs`
    :param batch_size: The batch size to split the data into
    :param num_epochs: The number of epochs to train on
    :param num_batches: How many batches to run on each epoch
    :return: The path to the fine-tuned model weights; which is the input to the evaluate workflow
    """
    # Load job configuration
    job_config = load_job_config(job_config_name)

    # Set the user_id
    job_config['user_id'] = user_id

    # Overwrite job config values with values from input, if any
    job_config['job_id'] = job_id = get_or_generate_job_id(job_config_name, job_id)
    if batch_size:
        job_config['batch_size'] = batch_size
    if num_epochs:
        job_config['num_epochs'] = num_epochs
    if num_batches:
        job_config['num_batches'] = num_batches

    # Instantiate the main logger
    logger = setup_logger('train_workflow', job_id, user_id)
    # Run the `train` workflow, and handle unexpected exceptions
    try:
        return run(dict(job_config), logger)
    except Exception as ex:
        logger.error(f"Exception occurred: {ex}")
        raise ex
