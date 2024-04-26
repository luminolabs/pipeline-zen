from unittest.mock import patch

import pytest

from common.dataset.provider.base import BaseDatasetProvider
from common.dataset.provider.huggingface import HuggingFace
from common.dataset.provider.utils import dataset_provider_factory


def test_base_dataset_provider():
    # Correct `dataset_id` and `split` types passed
    r = BaseDatasetProvider._validate_init(dataset_id='foo', split='bar')
    assert r is None

    # Incorrect `dataset_id` type passed
    with pytest.raises(TypeError):
        BaseDatasetProvider._validate_init(dataset_id=1, split='bar')

    # Incorrect `split` type passed
    with pytest.raises(TypeError):
        BaseDatasetProvider._validate_init(dataset_id='foo', split=1)

    # Downloads cache dir is a string
    assert BaseDatasetProvider.get_cache_dir() == \
           '../../.cache/datasets/basedatasetprovider'


@patch('common.dataset.provider.huggingface.load_dataset')
def test_huggingface_dataset(load_dataset,
                             huggingface_dataset, mock_dataset):
    # We mock the internally used function that fetches the dataset,
    # so this won't actually download anything.
    # `huggingface_dataset.fetch()` will call `load_dataset()` internally
    load_dataset.return_value = mock_dataset

    # Confirm `fetch()` method set the `dataset` property.. properly
    huggingface_dataset.fetch()
    assert huggingface_dataset.dataset == mock_dataset

    # Asking for an item returns the item
    assert huggingface_dataset[0] == mock_dataset[0]

    # Asking for the dataset length returns the length
    assert len(huggingface_dataset) == len(mock_dataset)


def test_dataset_provider_factory():
    # Invalid `dataset_provider` raises error
    with pytest.raises(TypeError):
        dataset_provider_factory(dataset_provider='foo',
                                 dataset_id='foo',
                                 split='bar')

    # Valid function arguments return a new object
    r = dataset_provider_factory(dataset_provider='huggingface',
                                 dataset_id='foo',
                                 split='bar')
    assert isinstance(r, HuggingFace)
