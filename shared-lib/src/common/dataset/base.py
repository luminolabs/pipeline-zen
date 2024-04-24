from abc import abstractmethod, ABC

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets
    """

    def __getitem__(self, item: int):
        """
        Get an item from the dataset.

        :param item: The index of the item
        :return: Returns the item from the dataset
        """
        return self._getitem(item)

    def __len__(self) -> int:
        """
        :return: Returns the length of the dataset
        """
        r = self._len()
        # Validate result is of type `int`
        if isinstance(r, int):
            return r
        raise TypeError('`Invalid `_len()` implementation. '
                        'The function must return an `int` value')

    @abstractmethod
    def _getitem(self, item: int):
        """
        :param item: The index of the item
        :return: A tuple of the input data and the label data
        """
        pass

    @abstractmethod
    def _len(self) -> int:
        """
        :return: Returns the number of elements in the dataset
        """
        pass
