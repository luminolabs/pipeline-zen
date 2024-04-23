from typing import Tuple, Optional

from common.preprocessor import text_transforms
from common.dataset.kind.input_label_dataset import InputLabelDataset


class TextTransformsDataset(InputLabelDataset):
    def __init__(self, dataset: InputLabelDataset,
                 transforms_input_func: str, transforms_label_func: Optional[str] = None):
        self.dataset = dataset
        self.transforms_input = getattr(text_transforms, transforms_input_func)
        self.transforms_label = \
            getattr(text_transforms, transforms_label_func) if transforms_label_func else None

    def __getitem__(self, item: int) -> Tuple[str, int]:
        text, label = self.dataset[item]
        transformed_text = self.transforms_input(text)
        transformed_label = self.transforms_label(label) \
            if self.transforms_label and isinstance(label, str) \
            else label
        return transformed_text, transformed_label

    def __len__(self) -> int:
        return len(self.dataset)
