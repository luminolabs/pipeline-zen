from types import ModuleType

from common.dataset.preprocessor.base import BaseDatasetPreprocessor
from common.preprocessor import text_transforms


class TextTransformsDataset(BaseDatasetPreprocessor):
    """
    Text transformations preprocessor.
    """

    @staticmethod
    def get_transforms_module() -> ModuleType:
        return text_transforms
