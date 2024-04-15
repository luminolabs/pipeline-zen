from abc import abstractmethod, ABC
from typing import Any


class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
