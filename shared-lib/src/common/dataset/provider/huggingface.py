import os
from typing import Any, Dict

from datasets import load_dataset

from common.dataset.provider.base import BaseDatasetProvider


class HuggingFace(BaseDatasetProvider):
    async def fetch(self, **kwargs):
        if kwargs.get('name'):
            print(f'... subset: `{kwargs.get("name")}`')
        self.dataset = load_dataset(
            path=self.dataset_id,
            keep_in_memory=False,
            cache_dir=os.path.join(self.get_cache_dir(), self.dataset_id),
            split=self.split,
            trust_remote_code=True,
            **kwargs
        )

    def __getitem__(self, item: int) -> Dict:
        return self.dataset[item]

    def __len__(self) -> int:
        return len(self.dataset)
