from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset:
    @abstractmethod
    async def fetch(self) -> None:
        """
        This should download the dataset on disk
        but not load it in memory
        """
        pass

    @abstractmethod
    async def to_torch_dataset(self, split: str) -> Dataset:
        """
        This should return an iterable torch dataset,
        and not load the whole dataset in memory;
        instead it should read the file X records at a time
        """
        pass
