from common.dataset.preprocessor.base import BaseDatasetPreprocessor
from common.dataset.preprocessor.torchvision_transforms import TorchvisionTransformsDataset
from common.dataset.preprocessor.text_transforms import TextTransformsDataset
from common.dataset.kind.single_label_dataset import SingleLabelDataset


def dataset_preprocess_factory(
        dataset_preprocess: str,
        dataset: SingleLabelDataset,
        **kwargs) -> BaseDatasetPreprocessor:
    """
    Factory method for creating a dataset preprocessor.

    :param dataset_preprocess: Short name of the preprocessor to use
    :param dataset: Dataset to iterate on; must be a `SingleLabelDataset`
    :param kwargs: Additional keyword arguments to pass to the dataset preprocessor
    :return: A preprocessor dataset
    """
    if dataset_preprocess == 'torchvision_transforms':
        return TorchvisionTransformsDataset(dataset, **kwargs)
    elif dataset_preprocess == 'text_transforms':
        return TextTransformsDataset(dataset, **kwargs)
