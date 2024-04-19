from abc import abstractmethod, ABC
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
