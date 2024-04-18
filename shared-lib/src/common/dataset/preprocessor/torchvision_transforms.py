from typing import Any

from common.preprocessor import torchvision_transforms
from common.dataset.kind.input_label_dataset import InputLabelDataset


class TorchvisionTransformsDataset(InputLabelDataset):
    def __init__(self, dataset: InputLabelDataset, transforms_func: str):
        self.dataset = dataset
        self.transforms_set = getattr(torchvision_transforms, transforms_func)()

    def __getitem__(self, item: int) -> Any:
        image, label = self.dataset[item]
        transformed_image = self.transforms_set(image)
        return transformed_image, label

    def __len__(self) -> int:
        return len(self.dataset)
