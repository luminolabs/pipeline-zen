from typing import Any

from torch.utils.data import Dataset

from common.tokenizer.utils import tokenizer_factory
from common.dataset.kind.value_label_dataset import ValueLabelDataset


class TokenizerDataset(ValueLabelDataset, Dataset):
    def __init__(self, dataset: ValueLabelDataset, tokenizer_name: str):
        self.dataset = dataset
        self.tokenizer = tokenizer_factory(tokenizer_name)

    def __getitem__(self, item: int) -> Any:
        value, label = self.dataset[item]
        tokenized_value = self.tokenizer(value)
        return tokenized_value, label

    def __len__(self) -> int:
        return len(self.dataset)
