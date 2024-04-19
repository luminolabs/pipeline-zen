import importlib
import os
import math
from typing import Tuple, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from common.dataset.kind.utils import dataset_kind_factory
from common.dataset.preprocessor.utils import dataset_preprocess_factory
from common.dataset.provider.utils import dataset_provider_factory
from common.model.utils import model_factory
from common.tokenizer.utils import tokenizer_factory


# This affects a few runtime options such as cache and results folders
# Available options so far: `docker`, `local`
environment = os.environ.get("ENVIRONMENT", 'local')


# Returns the path to a common root between workflows
# This allows workflows to share results, cache, etc
def get_root_path() -> str:
    if environment == 'local':
        return os.path.join('..', '..')
    elif environment == 'docker':
        return '.'


def load_job_config(job_config_id: str) -> dict:
    return importlib.import_module(f'job_configs.{job_config_id}').job_config


def get_results_path() -> str:
    path = os.path.join(get_root_path(), '.results')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_weights_path(job_id: Optional[str] = None) -> str:
    path_list = ()
    if job_id:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path_list = (job_id, timestamp + '.pt')
    path = os.path.join(get_results_path(), 'model_weights', *path_list)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


async def configure_model_and_dataloader(job_config: dict,
                                         for_inference: bool = False,
                                         model_weights_id: str = None) \
        -> Tuple[PreTrainedModel, DataLoader, PreTrainedTokenizerBase, str]:

    print("Loading and configuring dataset!")

    # Dataset split to use
    split = job_config.get('train_split')
    if for_inference:
        split = job_config.get('test_split')
    print(f'Using `{split}` split')
    # This is the dataset that pulls from the content provider
    # ex. huggingface, s3 providers
    dataset = dataset_provider_factory(
        dataset_provider=job_config.get('dataset_provider'),
        dataset_id=job_config.get('dataset_id'),
        split=split)
    dataset = await dataset.fetch()
    print(f'Dataset split has {len(dataset)} records')
    print(f'Batch size is {job_config.get("batch_size")}, '
          f'number of batches is {math.ceil(len(dataset)/job_config.get("batch_size"))}')
    if job_config.get('num_batches'):
        print(f'...but only {job_config.get("num_batches")} batches are configured to run')
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
    dataset_preprocess = dataset_preprocess_factory(
        dataset_preprocess=job_config.get('preprocessor'),
        dataset=dataset_kind,
        **job_config.get(job_config.get('preprocessor') + '_dataset_config'))

    # A tokenizer is used when we would like to convert text to tokens,
    # so that the text can be represented as an array of integers
    tokenizer = None
    if job_config.get('tokenizer_id'):
        tokenizer = tokenizer_factory(job_config.get('tokenizer_id'))

    # This loads data from the dataset in batches;
    # data requested from the dataloader will return preprocessed but not tokenized
    # Tokenization happens in the training loop, a batch at a time
    dataloader = DataLoader(
        dataset=dataset_preprocess,
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
    print("Training on (cpu/cuda/mps?) device:", device)

    print("Fetching the model")
    # Instantiate the appropriate model
    model = model_factory(
        model_kind=job_config.get('dataset_kind'),
        model_base=job_config.get('model_base'))
    if for_inference:
        model_weights_path = os.path.join(
            get_model_weights_path(), job_config.get('job_id'), model_weights_id)
        model.load_state_dict(torch.load(model_weights_path))
        print("Using model weights path: " + model_weights_path)
    model.to(device)
    if for_inference:
        model.eval()

    return model, dataloader, tokenizer, device


def tokenize_inputs(inputs, tokenizer: PreTrainedTokenizerBase, model_args: dict, device):
    # Tokenize batch of inputs
    # Tensor data need to be of same length, so we need to
    # set max size and padding options
    tokenized_values = tokenizer(
        inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.ONLY_FIRST,
        max_length=tokenizer.model_max_length,
        return_tensors=TensorType.PYTORCH)
    # Replace original inputs with tokenized inputs
    inputs = tokenized_values.get('input_ids')
    # Load attention masks to device
    attention_masks = tokenized_values.get('attention_mask').to(device)
    model_args['attention_mask'] = attention_masks
    return inputs
