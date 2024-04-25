import pytest

from common.dataset.kind.utils import dataset_kind_factory
from common.dataset.preprocessor.utils import dataset_preprocessor_factory
from common.dataset.provider.utils import dataset_provider_factory


@pytest.fixture
def huggingface_dataset():
    return dataset_provider_factory(
        dataset_provider='huggingface',
        dataset_id='foo', split='bar')


@pytest.fixture
def single_label_dataset(huggingface_dataset):
    return dataset_kind_factory(
        dataset_kind='single_label',
        dataset=huggingface_dataset,
        input_col='input',
        label_col='label')


@pytest.fixture
def text_transforms_dataset(single_label_dataset):
    return dataset_preprocessor_factory(
        dataset_preprocessor='text_transforms',
        dataset=single_label_dataset,
        transforms_input_func='strip',
        transforms_label_func='strip')


@pytest.fixture
def torchvision_transforms_dataset(single_label_dataset):
    return dataset_preprocessor_factory(
        dataset_preprocessor='torchvision_transforms',
        dataset=single_label_dataset,
        transforms_input_func='to_tensor',
        transforms_label_func='to_tensor')


@pytest.fixture
def mock_dataset_item():
    return {'input': 'in', 'label': 'out'}


@pytest.fixture
def mock_dataset(mock_dataset_item):
    return [mock_dataset_item]
