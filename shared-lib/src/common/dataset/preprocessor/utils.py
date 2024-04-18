from typing import Union

from common.dataset.preprocessor.torchvision_transforms import TorchvisionTransformsDataset
from common.dataset.preprocessor.text_transforms import TextTransformsDataset
from common.dataset.kind.input_label_dataset import InputLabelDataset


def dataset_preprocess_factory(
        dataset_preprocess: str,
        dataset: Union[InputLabelDataset,],
        **kwargs) -> Union[TorchvisionTransformsDataset, TextTransformsDataset,]:
    if dataset_preprocess == 'torchvision_transforms':
        return TorchvisionTransformsDataset(dataset, **kwargs)
    elif dataset_preprocess == 'text_transforms':
        return TextTransformsDataset(dataset, **kwargs)
