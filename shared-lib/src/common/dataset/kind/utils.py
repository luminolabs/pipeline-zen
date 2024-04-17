from typing import Union

from common.dataset.base import BaseDataset
from common.dataset.kind.value_label_dataset import ValueLabelDataset


def dataset_kind_factory(dataset_kind: str, dataset: BaseDataset, **kwargs) -> Union[ValueLabelDataset,]:
    if dataset_kind == 'value_label':
        return ValueLabelDataset(dataset, **kwargs)
    if dataset_kind == '...':
        # TODO: Implement dataset for other kind of model training, ex. LLM
        pass
