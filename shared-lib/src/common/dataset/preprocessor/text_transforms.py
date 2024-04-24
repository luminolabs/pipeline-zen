from typing import Any, Tuple

from common.preprocessor import text_transforms
from common.dataset.kind.input_label_dataset import InputLabelDataset


class TextTransformsDataset(InputLabelDataset):
    def __init__(self, dataset: InputLabelDataset, transforms_input_func: str):
        self.dataset = dataset
        self.transforms_input = getattr(text_transforms, transforms_input_func)

    def __getitem__(self, item: int) -> Tuple[str, int]:
        text, label = self.dataset[item]
        transformed_text = self.transforms_input(text)
        return transformed_text, label

    def __len__(self) -> int:
        return len(self.dataset)
