from unittest.mock import patch

import pytest
from PIL import Image
from torch import Tensor

from common.dataset.preprocessor.base import BaseDatasetPreprocessor
from common.dataset.preprocessor.text_transforms import TextTransformsDataset
from common.dataset.preprocessor.utils import dataset_preprocessor_factory
from common.preprocessor import text_transforms, torchvision_transforms


def test_base_dataset_preprocessor(single_label_dataset, huggingface_dataset):
    # Correct `dataset` and `transforms_input_*` types passed, no raised error
    BaseDatasetPreprocessor._validate_init(dataset=single_label_dataset,
                                           transforms_input_func='func1',
                                           transforms_label_func='func2')

    # `transforms_label_func` is optional, no raised error
    BaseDatasetPreprocessor._validate_init(dataset=single_label_dataset,
                                           transforms_input_func='func1',
                                           transforms_label_func='func2')

    # `transforms_input_func` must be a string, not a callable
    with pytest.raises(TypeError):
        BaseDatasetPreprocessor._validate_init(dataset=single_label_dataset,
                                               transforms_input_func=lambda x: x,
                                               transforms_label_func='func2')

    # `transforms_label_func` must be a string, not a callable
    with pytest.raises(TypeError):
        BaseDatasetPreprocessor._validate_init(dataset=single_label_dataset,
                                               transforms_input_func='func1',
                                               transforms_label_func=lambda x: x)


@patch('common.dataset.provider.huggingface.load_dataset')
def test_text_transforms_dataset(load_dataset, text_transforms_dataset, logger):
    # Dataset that requires preprocessing (trim whitespace)
    dataset = [{'input': '   in   ', 'label': '   out   '}]
    # Configure provider dataset with mock data from hugging face
    load_dataset.return_value = dataset
    text_transforms_dataset.dataset.dataset.fetch(logger)
    # Confirm that `TextTransformsDataset` properly preprocesses items
    # Note that `text_transforms_dataset` is a fixture that is
    # configured to use the `strip()` preprocessor function
    assert text_transforms_dataset[0] == ('in', 'out')

    # Confirm that `_get_transforms_module() returns the
    # correct transforms module
    assert text_transforms_dataset._get_transforms_module() == text_transforms

    # Dataset that's the wrong type (ie. not text)
    dataset = [{'input': 0, 'label': 'out'}]
    # Configure provider dataset with mock data from hugging face
    load_dataset.return_value = dataset
    text_transforms_dataset.dataset.dataset.fetch(logger)
    # If the input column datatype is not a string, then this dataset
    # must raise error
    with pytest.raises(AttributeError):
        _ = text_transforms_dataset[0]


@patch('common.dataset.provider.huggingface.load_dataset')
def test_torchvision_transforms_dataset(load_dataset, torchvision_transforms_dataset, logger):
    # Configure images to run tests with
    image_in = Image.new('L', (1, 1))
    image_out = Image.new('L', (1, 1))

    # Dataset with image input, string output
    dataset = [{'input': image_in, 'label': image_out}]
    # Configure provider dataset with mock data from hugging face
    load_dataset.return_value = dataset
    torchvision_transforms_dataset.dataset.dataset.fetch(logger)
    # Confirm that `torchvision_transforms_dataset` properly preprocesses items
    # Note that `torchvision_transforms_dataset` is a fixture that is
    # configured to use the `to_tensor()` preprocessor function
    # Assert we went from PIL image to Tensor
    assert len(torchvision_transforms_dataset[0]) == 2
    assert isinstance(torchvision_transforms_dataset[0][0], Tensor)
    assert isinstance(torchvision_transforms_dataset[0][1], Tensor)

    # Confirm that `_get_transforms_module() returns the
    # correct transforms module
    assert torchvision_transforms_dataset._get_transforms_module() == torchvision_transforms

    # Dataset that's the wrong type (ie. not `PIL.Image.Image`)
    dataset = [{'input': 0, 'label': 'out'}]
    # Configure provider dataset with mock data from hugging face
    load_dataset.return_value = dataset
    torchvision_transforms_dataset.dataset.dataset.fetch(logger)
    # If the input column datatype is not a string, then this dataset
    # must raise error
    with pytest.raises(AttributeError):
        _ = torchvision_transforms_dataset[0]


def test_dataset_preprocessor_factory(single_label_dataset):
    # Invalid `dataset_provider` raises error
    with pytest.raises(TypeError):
        dataset_preprocessor_factory(dataset_preprocessor='foo',
                                     dataset=single_label_dataset,
                                     transforms_input_func='strip')

    # Valid function arguments return a new object
    r = dataset_preprocessor_factory(dataset_preprocessor='text_transforms',
                                     dataset=single_label_dataset,
                                     transforms_input_func='strip')
    assert isinstance(r, TextTransformsDataset)
