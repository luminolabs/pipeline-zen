from abc import ABC, abstractmethod
from typing import Tuple, Any

from common.dataset.base import BaseDataset
from common.dataset.provider.base import BaseDatasetProvider


class BaseDatasetKind(BaseDataset, ABC):
    """
    Base class for all kinds of datasets
    """

    def __init__(self, dataset: BaseDataset, input_col: str):
        """
        :param dataset: The dataset to work with. It has to be of type `BaseDatasetProvider`
        :param input_col: The name of the input key (ex. `image`)
        """
        if not isinstance(dataset, BaseDatasetProvider):
            raise TypeError('`dataset` must be of type `BaseDatasetProvider`')
        if not isinstance(input_col, str):
            raise TypeError('`input_col` must be of type `str`')

        self.dataset = dataset
        self.input_col = input_col

    @staticmethod
    @abstractmethod
    def _num_labels() -> int:
        """
        :return: Returns the number of labels in the dataset; used to run validation rules
        """
        pass

    def __getitem__(self, item: int):
        r = self._getitem(item)
        # Validate correct number of items was returned
        num_elements_to_return = 1 + self._num_labels()
        if isinstance(r, tuple) and len(r) == num_elements_to_return:
            return r
        raise TypeError(f'`Invalid `_getitem()` implementation. '
                        f'The function must return a tuple with at least `{num_elements_to_return}` elements.')

    def _len(self) -> int:
        return len(self.dataset)
