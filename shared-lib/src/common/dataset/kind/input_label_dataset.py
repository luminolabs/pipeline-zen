from typing import Any, Tuple

from common.dataset.base import BaseDataset


class InputLabelDataset(BaseDataset):
    def __init__(self, dataset: BaseDataset, input_col: str, label_col: str):
        self.dataset = dataset
        self.input_col = input_col
        self.label_col = label_col

    def __getitem__(self, item: int) -> Tuple[Any, Any]:
        return self.dataset[item][self.input_col], self.dataset[item][self.label_col]

    def __len__(self) -> int:
        return len(self.dataset)
