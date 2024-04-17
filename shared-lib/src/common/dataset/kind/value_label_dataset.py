from typing import Any

from torch.utils.data import Dataset

from common.dataset.base import BaseDataset


class ValueLabelDataset(BaseDataset, Dataset):
    def __init__(self, dataset: BaseDataset, value_col: str, label_col: str):
        self.dataset = dataset
        self.value_col = value_col
        self.label_col = label_col

    def __getitem__(self, item: int) -> Any:
        return self.dataset[item][self.value_col], self.dataset[item][self.label_col]

    def __len__(self) -> int:
        return len(self.dataset)
