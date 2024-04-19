from typing import Union

from common.dataset.base import BaseDataset
from common.dataset.kind.input_label_dataset import InputLabelDataset


def dataset_kind_factory(dataset_kind: str, dataset: BaseDataset, **kwargs) -> Union[InputLabelDataset,]:
    if dataset_kind == 'input_label':
        return InputLabelDataset(dataset, **kwargs)
    if dataset_kind == '...':
        # TODO: Implement dataset for other kind of model training, ex. LLM
        pass
