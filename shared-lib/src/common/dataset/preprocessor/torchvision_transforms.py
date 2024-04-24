from typing import Tuple, Union, Optional

from torch import Tensor

from common.preprocessor import torchvision_transforms
from common.dataset.kind.input_label_dataset import InputLabelDataset


class TorchvisionTransformsDataset(InputLabelDataset):
    def __init__(self,
                 dataset: InputLabelDataset,
                 transforms_input_func: str, transforms_label_func: Optional[str] = None):
        self.dataset = dataset
        self.transforms_input = getattr(torchvision_transforms, transforms_input_func)()
        self.transforms_label = \
            getattr(torchvision_transforms, transforms_label_func)() if transforms_label_func \
            else None

    def __getitem__(self, item: int) -> Tuple[Tensor, Union[int, Tensor]]:
        image, label = self.dataset[item]
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transforms_input(image)
        transformed_label = self.transforms_label(label) \
            if self.transforms_label else label
        return transformed_image, transformed_label

    def __len__(self) -> int:
        return len(self.dataset)
