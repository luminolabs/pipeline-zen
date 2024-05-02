import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from common.helpers import configure_model_and_dataloader
from common.utils import load_job_config


def test_configure_model_and_dataloader(logger):
    # Test that `configure_model_and_dataloader` initializes
    # objects properly
    job_config = load_job_config('imdb_sentiment')
    for_inference = False
    model_weights_id = None
    model, dataloader, tokenizer, device = \
        configure_model_and_dataloader(job_config, logger, for_inference, model_weights_id)
    assert isinstance(model, PreTrainedModel)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert isinstance(device, torch.device)

    # If `model_weights_id` doesn't point to a valid
    # weights file, raise error
    for_inference = True
    model_weights_id = 'foo'
    with pytest.raises(FileNotFoundError):
        configure_model_and_dataloader(job_config, logger, for_inference, model_weights_id)