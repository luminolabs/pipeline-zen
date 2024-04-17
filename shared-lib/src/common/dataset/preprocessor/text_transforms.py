from typing import Any

from torch.utils.data import Dataset

from common.preprocessor import text_transforms
from common.dataset.kind.value_label_dataset import ValueLabelDataset


class TextTransformsDataset(ValueLabelDataset, Dataset):
    def __init__(self, dataset: ValueLabelDataset, transforms_func: str):
        self.dataset = dataset
        self.transforms_set = getattr(text_transforms, transforms_func)

    def __getitem__(self, item: int) -> Any:
        text, label = self.dataset[item]
        transformed_text = self.transforms_set(text)
        return transformed_text, label

    def __len__(self) -> int:
        return len(self.dataset)
