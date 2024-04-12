from datasets import load_dataset
from torch.utils.data import Dataset

from dataset.base_dataset import BaseDataset


class HuggingFace(BaseDataset):
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path

    async def fetch(self) -> None:
        # TODO: Is there support for async download?
        load_dataset(
            path=self.dataset_path,
            keep_in_memory=False
        )

    async def to_torch_dataset(self, split: str) -> Dataset:
        # TODO
        pass
