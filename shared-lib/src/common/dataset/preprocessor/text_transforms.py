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

    def _getitem(self, item: int):
        input, label = super()._getitem(item)

        if not isinstance(input, str):
            raise AttributeError('This class must only be used with `str` inputs')

        return input, label
