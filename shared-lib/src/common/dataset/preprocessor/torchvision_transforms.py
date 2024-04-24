from types import ModuleType

from common.dataset.preprocessor.base import BaseDatasetPreprocessor
from common.preprocessor import torchvision_transforms


class TorchvisionTransformsDataset(BaseDatasetPreprocessor):
    """
    Image transformations preprocessor.
    """

    @staticmethod
    def get_transforms_module() -> ModuleType:
        return torchvision_transforms

    def _getitem(self, item: int):
        input, label = super()._getitem(item)
        # If input is a two-dimensional image, convert to three-dimensional
        if input.mode == 'L':
            input = input.convert('RGB')
        return input, label
