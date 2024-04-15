from typing import Union

from common.dataset.base import BaseDataset
from common.dataset.kind.image_dataset import ImageDataset


def dataset_kind_factory(dataset_kind: str, dataset: BaseDataset, **kwargs) -> Union[ImageDataset,]:
    if dataset_kind == 'image':
        return ImageDataset(dataset, **kwargs)
    if dataset_kind == '...':
        # TODO: Implement dataset for other kind of model training, ex. LLM
        pass
