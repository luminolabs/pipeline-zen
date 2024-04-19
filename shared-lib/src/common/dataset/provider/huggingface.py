import os
from typing import Any

from datasets import load_dataset

from common.dataset.provider.base import BaseDatasetProvider


class HuggingFace(BaseDatasetProvider):
    async def fetch(self) -> Any:
        self.dataset = load_dataset(
            path=self.dataset_id,
            keep_in_memory=False,
            cache_dir=os.path.join(self.get_cache_dir(), self.dataset_id),
            split=self.split
        )
        return self.dataset

    def __getitem__(self, item: int) -> Any:
        return self.dataset[self.split][item]

    def __len__(self) -> int:
        return len(self.dataset[self.split])
