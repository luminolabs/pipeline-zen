from typing import Union

from common.dataset.preprocessor.torchvision_transforms import TorchvisionTransformsDataset
from common.dataset.kind.image_dataset import ImageDataset


def dataset_preprocess_factory(
        dataset_preprocess: str,
        dataset: Union[ImageDataset,],
        **kwargs) -> Union[TorchvisionTransformsDataset,]:
    if dataset_preprocess == 'torchvision_transforms':
        return TorchvisionTransformsDataset(dataset, **kwargs)
    if dataset_preprocess == '...':
        # TODO: Implement preprocessors for other kind of model training, ex. LLM
        pass
