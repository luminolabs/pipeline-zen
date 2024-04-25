from abc import ABC, abstractmethod
from types import ModuleType
from typing import Optional, Tuple, Union, Any

from torch import Tensor

from common.dataset.base import BaseDataset
from common.dataset.kind.single_label_dataset import SingleLabelDataset


class BaseDatasetPreprocessor(BaseDataset, ABC):
    """
    Abstract base class for preprocessing datasets.
    """

    def __init__(self,
                 dataset: SingleLabelDataset,
                 transforms_input_func: str, transforms_label_func: Optional[str] = None):
        """
        Transform functions are imported from the module specified by
        `get_transforms_module()` under each subclass. Transform functions
        are usually located under `common.preprocessor`

        :param dataset: Dataset to be preprocessed.
        :param transforms_input_func: Function which transforms input data.
        :param transforms_label_func: Function which transforms label data.
        """

        if not isinstance(dataset, SingleLabelDataset):
            raise TypeError('`dataset` must be of type `SingleLabelDataset`')
        if not isinstance(transforms_input_func, str):
            raise TypeError('`transforms_input_func` must be of type `str`')
        if transforms_label_func and not isinstance(transforms_label_func, str):
            raise TypeError('`transforms_label_func` must be of type `str`')

        self.dataset = dataset
        self.transforms_input = getattr(self.get_transforms_module(), transforms_input_func)
        self.transforms_label = \
            getattr(self.get_transforms_module(), transforms_label_func) \
                if transforms_label_func \
                else None

    def __getitem__(self, item: int) -> Tuple[Union[int, str, Tensor], Union[int, str, Tensor]]:
        """
        Run transformations on input and label data.
        :param item: Item to be transformed.
        :return: Transformed input and label data.
        """
        input, label = self._getitem(item)
        transformed_input = self.transforms_input(input) if self.transforms_input else input
        transformed_label = self.transforms_label(label) if self.transforms_label else label
        return transformed_input, transformed_label

    def _getitem(self, item: int) -> Tuple[Any, Any]:
        return self.dataset[item]

    def _len(self):
        return len(self.dataset)

    @staticmethod
    @abstractmethod
    def get_transforms_module() -> ModuleType:
        """
        Returns the transforms module.
        """
        pass
