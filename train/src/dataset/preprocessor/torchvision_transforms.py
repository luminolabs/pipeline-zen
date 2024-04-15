from typing import Any

from torch.utils.data import Dataset
from preprocessor import torchvision_transforms

from dataset.provider.base import BaseDataset
from dataset.kind.image_dataset import ImageDataset


class TorchvisionTransformsDataset(BaseDataset, Dataset):
    def __init__(self, dataset: ImageDataset, transforms_func: str):
        self.dataset = dataset
        self.transforms_set = getattr(torchvision_transforms, transforms_func)()

    def __getitem__(self, item: int) -> Any:
        image, label = self.dataset[item]
        transformed_image = self.transforms_set(image)
        return transformed_image, label

    def __len__(self) -> int:
        return len(self.dataset)