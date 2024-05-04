from abc import ABC, abstractmethod

from common.dataset.base import BaseDataset
from common.dataset.provider.base import BaseDatasetProvider


class BaseDatasetKind(BaseDataset, ABC):
    """
    Base class for all kinds of datasets
    """

    def __init__(self, dataset: BaseDatasetProvider, input_col: str):
        """
        :param dataset: The dataset to work with. It has to be of type `BaseDatasetProvider`
        :param input_col: The name of the input key (ex. `image`)
        """
        self._validate_init(dataset, input_col)
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
        value = self._getitem(item)
        num_elements_to_return = 1 + self._num_labels()
        self._validate_get_item(value, num_elements_to_return)
        return value

    def _len(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _validate_init(dataset: BaseDatasetProvider, input_col: str):
        """
        This was separated from `__init__()` so that it can be unit tested
        """
        if not isinstance(dataset, BaseDatasetProvider):
            raise TypeError('`dataset` must be of type `BaseDatasetProvider`')
        if not isinstance(input_col, str):
            raise TypeError('`input_col` must be of type `str`')

    @staticmethod
    def _validate_get_item(value: tuple, num_elements_to_return: int):
        """
        This was separated from `__getitem__()` so that it can be unit tested
        """
        # Validate correct number of elements in item
        if not isinstance(value, tuple) or len(value) != num_elements_to_return:
            raise TypeError(f'Invalid `_getitem()` implementation. '
                            f'The function must return a tuple with at least `{num_elements_to_return}` elements.')