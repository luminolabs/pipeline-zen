from unittest.mock import patch

import pytest

from common.dataset.kind.base import BaseDatasetKind
from common.dataset.kind.single_label_dataset import SingleLabelDataset
from common.dataset.kind.utils import dataset_kind_factory
from common.dataset.provider.huggingface import HuggingFace


def test_base_dataset_kind(huggingface_dataset, text_transforms_dataset):
    # Correct `dataset` and `input_col` types passed
    r = BaseDatasetKind._validate_init(dataset=huggingface_dataset, input_col='input')
    assert r is None

    # Incorrect `input_col` type passed
    with pytest.raises(TypeError):
        BaseDatasetKind._validate_init(dataset=huggingface_dataset, input_col=0)

    # Incorrect `dataset` type passed
    with pytest.raises(TypeError):
        BaseDatasetKind._validate_init(dataset=text_transforms_dataset, input_col='input')


@patch.object(HuggingFace, '_getitem')
def test_single_label_dataset(
        _getitem,
        huggingface_dataset, mock_dataset_item, mock_dataset_item_with_master_col):
    # Correct `label_col` type passed
    r = SingleLabelDataset(dataset=huggingface_dataset,
                           input_col='input',
                           label_col='label')
    assert isinstance(r, SingleLabelDataset)

    # Incorrect `label_col` type passed
    with pytest.raises(TypeError):
        SingleLabelDataset(dataset=huggingface_dataset,
                           input_col='input',
                           label_col=1)

    # Incorrect `master_col` type passed
    with pytest.raises(TypeError):
        SingleLabelDataset(dataset=huggingface_dataset,
                           master_col='master',
                           label_col=1)

    # Correct number of labels is set
    assert r._num_labels() == 1

    # Asking for an item returns a tuple with 2 elements
    # and the right values
    _getitem.return_value = mock_dataset_item
    # ex: (1, 2) == (1, 2)
    assert r[0] == tuple([v for _, v in mock_dataset_item.items()])

    # Same as above, but emulate dataset with a single col
    _getitem.return_value = mock_dataset_item_with_master_col
    r.master_col = 'master'
    assert r[0] == tuple([v for _, v in mock_dataset_item.items()])



def test_dataset_kind_factory(huggingface_dataset):
    # Invalid `dataset_kind` raises error
    with pytest.raises(TypeError):
        dataset_kind_factory(dataset_kind='foo',
                             dataset=huggingface_dataset,
                             input_col='input')

    # Valid function arguments return a new object
    r = dataset_kind_factory(dataset_kind='single_label',
                             dataset=huggingface_dataset,
                             input_col='input',
                             label_col='label')
    assert isinstance(r, SingleLabelDataset)
