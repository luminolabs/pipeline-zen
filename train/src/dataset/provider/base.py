import os
from abc import abstractmethod
from typing import Any

from dataset.base import BaseDataset


class BaseDatasetProvider(BaseDataset):
    def __init__(self, dataset_path: str, split: str) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.dataset = None

    @abstractmethod
    async def fetch(self) -> Any:
        """
        This should download the dataset on disk
        but not load it in memory
        """
        pass

    def get_cache_dir(self):
        """
        ex: `.cache/huggingface/datasets`
        The dataset name is appended to the path above in the subclasses
        """
        return os.path.join('.cache', self.__class__.__name__.lower(), 'datasets')
