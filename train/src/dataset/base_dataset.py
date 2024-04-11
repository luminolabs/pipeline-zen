from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset:
    @abstractmethod
    async def load(self) -> None:
        pass

    @abstractmethod
    async def to_torch_dataset(self, split: str) -> Dataset:
        pass
