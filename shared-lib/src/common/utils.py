import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from common.dataset.kind.utils import dataset_kind_factory
from common.dataset.preprocessor.utils import dataset_preprocess_factory
from common.dataset.provider.utils import dataset_provider_factory
from common.model.utils import model_factory
from common.tokenizer.utils import tokenizer_factory


async def configure_model_and_dataloader(job_config: dict, for_inference: bool = False) \
        -> Tuple[PreTrainedModel, DataLoader, PreTrainedTokenizerBase, str]:

    print("Loading and configuring dataset!")
    # This is the dataset that pulls from the content provider
    # ex. huggingface, s3 providers
    dataset = dataset_provider_factory(
        dataset_provider=job_config.get('dataset_provider'),
        dataset_id=job_config.get('dataset_id'),
        split=job_config.get('train_split'))
    dataset = await dataset.fetch()
    # This is the dataset that prepares the dataset data into the data structre
    # that will be used in the training loop
    # ex. the `input_label` dataset converts data structured as an arbitrary dict to tuple(input, label)
    dataset_kind = dataset_kind_factory(
        dataset_kind=job_config.get('dataset_kind'),
        dataset=dataset,
        **job_config.get(job_config.get('dataset_kind') + '_dataset_config'))
    # This is the preprocessing dataset,
    # it will apply transformations and prepare data for training
    # ex. `text_transforms` can remove whitespaces, usernames, etc from the input string
    dataset_preprocess = dataset_preprocess_factory(
        dataset_preprocess=job_config.get('preprocessor'),
        dataset=dataset_kind,
        **job_config.get(job_config.get('preprocessor') + '_dataset_config'))

    # A tokenizer is used when we would like to convert text to tokens,
    # so that the text can be represented as an array of integers
    tokenizer = None
    if 'tokenizer_id' in job_config:
        tokenizer = tokenizer_factory(job_config.get('tokenizer_id'))

    # This loads data from the dataset in batches;
    # data requested from the dataloader will return preprocessed but not tokenized
    # Tokenization happens in the training loop, a batch at a time
    dataloader = DataLoader(
        dataset_preprocess,
        batch_size=job_config.get('batch_size'),
        shuffle=job_config.get('shuffle'))

    # To run on GPU or not to run on GPU, that is the question
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # `mps` device enables high-performance training on GPU for MacOS devices with Metal programming framework.
        # see: https://pytorch.org/docs/stable/notes/mps.html
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        device = 'mps'
    device = torch.device(device)
    print("Training on (CPU/GPU/MPS?) device:", device)

    print("Fetching the model")
    # Instantiate the appropriate model
    model = model_factory(
        model_kind=job_config.get('dataset_kind'),
        model_base=job_config.get('model_base'))
    if for_inference:
        model.load_state_dict(torch.load(job_config.get('model_weights_path')))
    model.to(device)
    if for_inference:
        model.eval()

    return model, dataloader, tokenizer, device
