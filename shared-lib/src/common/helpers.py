import math
import os
from logging import Logger
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from common.dataset.kind.utils import dataset_kind_factory
from common.dataset.preprocessor.utils import dataset_preprocessor_factory
from common.dataset.provider.utils import dataset_provider_factory
from common.model.utils import model_factory
from common.tokenizer.utils import tokenizer_factory
from common.utils import get_model_weights_path


def get_device(logger: Logger):
    """
    Returns a torch device with the following priority (highest to lowest):
    cuda -> mps -> cpu

    :param logger: The logger instance
    :return: Torch device
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # `mps` device enables high-performance training on GPU for MacOS devices with Metal programming framework.
        # see: https://pytorch.org/docs/stable/notes/mps.html
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        device = 'mps'
    logger.info('Training on (cpu/cuda/mps?) device: ' + device)
    device = torch.device(device)
    return device


def configure_model_and_dataloader(job_config: dict,
                                   logger: Logger,
                                   for_inference: bool = False,
                                   model_weights_id: str = None) \
        -> Tuple[PreTrainedModel, DataLoader, PreTrainedTokenizerBase, str]:
    """
    Configure model and dataloader from a job configuration.

    :param job_config: The job configuration. Configurations are found under `job_configs`
    :param logger: The logger instance
    :param for_inference: Whether we are running inference or training
    :param model_weights_id: Model weights to use for inference
    :return: Configured objects to be used in the workflow
    """
    if for_inference:
        model_weights_path = os.path.join(
            get_model_weights_path(), job_config.get('job_id'), model_weights_id)
        if not os.path.isfile(model_weights_path):
            raise FileNotFoundError(f'model_weights_id: {model_weights_id} '
                                    f'not found; looked at: {model_weights_path}')

    logger.info("Loading and configuring dataset!")

    # Dataset split to use
    split = job_config.get('train_split')
    if for_inference:
        split = job_config.get('test_split')
    # This is the dataset that pulls from the content provider
    # ex. huggingface, s3 providers
    dataset = dataset_provider_factory(
        dataset_provider=job_config.get('dataset_provider'),
        dataset_id=job_config.get('dataset_id'),
        split=split,
        logger=logger)
    dataset.fetch(logger, **job_config.get('dataset_fetch_config', {}))
    logger.info(f'Dataset split has {len(dataset)} records')
    logger.info(f'Batch size is {job_config.get("batch_size")}, '
          f'number of batches is {math.ceil(len(dataset) / job_config.get("batch_size"))}')
    if job_config.get('num_batches'):
        logger.info(f'...but only {job_config.get("num_batches")} batches are configured to run')
    # This is the dataset that prepares the dataset data into the data structure
    # that will be used in the training loop
    # ex. the `input_label` dataset converts data structured as an arbitrary dict to tuple(input, label)
    dataset_kind = dataset_kind_factory(
        dataset_kind=job_config.get('dataset_kind'),
        dataset=dataset,
        **job_config.get(job_config.get('dataset_kind') + '_dataset_config'))
    # This is the preprocessing dataset,
    # it will apply transformations and prepare data for training
    # ex. `text_transforms` can remove whitespaces, usernames, etc from the input string
    dataset_preprocessor = None
    if job_config.get('preprocessor'):
        dataset_preprocessor = dataset_preprocessor_factory(
            dataset_preprocessor=job_config.get('preprocessor'),
            dataset=dataset_kind,
            **job_config.get(job_config.get('preprocessor') + '_dataset_config'))

    # A tokenizer is used when we would like to convert text to tokens,
    # so that the text can be represented as an array of integers
    tokenizer = None
    if job_config.get('tokenizer_id'):
        tokenizer = tokenizer_factory(job_config.get('tokenizer_id'), logger)

    # This loads data from the dataset in batches;
    # data requested from the dataloader will return preprocessed but not tokenized
    # Tokenization happens in the training loop, a batch at a time
    dataloader = DataLoader(
        dataset=dataset_preprocessor if dataset_preprocessor else dataset_kind,
        batch_size=job_config.get('batch_size'),
        shuffle=job_config.get('shuffle'))

    # To run on GPU or not to run on GPU, that is the question
    device = get_device(logger)

    logger.info("Fetching the model")
    # Instantiate the appropriate model
    model = model_factory(
        model_kind=job_config.get('dataset_kind'),
        model_base=job_config.get('model_base'),
        logger=logger,
        **job_config.get('model_base_args', {}))
    if for_inference:
        # `map_location` is needed when the weights were generated on
        # a different kind of a device
        # ex. `cpu` running on this machine vs weights generated with `cuda`
        model_weights = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(model_weights)
        logger.info("Using model weights path: " + model_weights_path)
    model.to(device)
    if for_inference:
        model.eval()
    elif torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        # Wrap the model with DataParallel
        model = nn.DataParallel(model)

    return model, dataloader, tokenizer, device
