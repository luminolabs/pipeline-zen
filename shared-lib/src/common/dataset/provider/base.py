import os
from abc import abstractmethod
from logging import Logger
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
        self._validate_init(dataset_id, split)

        self.dataset_id = dataset_id
        self.split = split
        self.dataset = None  # To be set in subclasses

    @abstractmethod
    def fetch(self, logger: Logger, **kwargs) -> Any:
        """
        :param logger: The logger instance

        This should download the dataset on disk
        but not load it in memory
        """
        pass

    @classmethod
    def get_cache_dir(cls):
        """
        Where to store the dataset locally
        ex: `.cache/huggingface/datasets`

        The dataset name is appended to the path above in the subclasses
        """
        return os.path.join(utils.get_root_path(), '.cache', 'datasets', cls.__name__.lower())

    @staticmethod
    def _validate_init(dataset_id: str, split: str):
        """
        This was separated from `__init__()` so that it can be unit tested
        """
        if not isinstance(dataset_id, str):
            raise TypeError('`dataset_id` must be of type `str`')
        if not isinstance(split, str):
            raise TypeError('`split` must be of type `str`')
