import os
from abc import abstractmethod
from typing import Any

from common import utils
from common.dataset.base import BaseDataset


class BaseDatasetProvider(BaseDataset):
    """
    Abstract base class for dataset providers, such as huggingface or s3
    """

    def __init__(self, dataset_id: str, split: str) -> None:
        """
        :param dataset_id: The dataset ID to download
        :param split: The dataset split to download (ex. train, valid, test)
        """
        self.dataset_id = dataset_id
        self.split = split

    @abstractmethod
    async def fetch(self, **kwargs) -> Any:
        """
        This should download the dataset on disk
        but not load it in memory
        """
        pass

    def get_cache_dir(self):
        """
        Where to store the dataset locally
        ex: `.cache/huggingface/datasets`

        The dataset name is appended to the path above in the subclasses
        """
        return os.path.join(utils.get_root_path(), '.cache', 'datasets', self.__class__.__name__.lower())
