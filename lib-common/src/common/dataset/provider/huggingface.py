import os
from logging import Logger
from typing import Dict

from datasets import load_dataset

from common.dataset.provider.base import BaseDatasetProvider


class HuggingFace(BaseDatasetProvider):
    """
    HuggingFace Dataset Provider
    """

    def fetch(self, logger: Logger, **kwargs):
        # Some huggingface datasets consist of multiple subsets;
        # use the `name` kwarg to specify which to use
        if kwargs.get('name'):
            logger.info(f'... subset: `{kwargs.get("name")}`')
        # Download the dataset
        # `load_dataset()` has builtin retries (1 retry by default)
        self.dataset = load_dataset(
            path=self.dataset_id,
            # Only store in disk
            keep_in_memory=False,
            cache_dir=os.path.join(self.get_dataset_cache_dir(), self.dataset_id),
            split=self.split,
            # Some datasets require running a script
            # to prepare data upon download
            trust_remote_code=True,
            **kwargs
        )

    def _getitem(self, item: int) -> Dict:
        """
        :return: Returns dictionary with column names as keys
        """
        return self.dataset[item]

    def _len(self) -> int:
        return len(self.dataset)
