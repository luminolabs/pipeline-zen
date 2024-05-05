from common.dataset.preprocessor.torchvision_transforms import TorchvisionTransformsDataset
from common.dataset.preprocessor.text_transforms import TextTransformsDataset
from common.dataset.kind.single_label_dataset import SingleLabelDataset


def dataset_preprocessor_factory(
        dataset_preprocessor: str,
        dataset: SingleLabelDataset,
        **kwargs):
    """
    Factory method for creating a dataset preprocessor.

    :param dataset_preprocessor: Short name of the preprocessor to use
    :param dataset: Dataset to iterate on; must be a `SingleLabelDataset`
    :param kwargs: Additional keyword arguments to pass to the dataset preprocessor
    :return: A preprocessor dataset
    """
    if dataset_preprocessor == 'torchvision_transforms':
        return TorchvisionTransformsDataset(dataset, **kwargs)
    elif dataset_preprocessor == 'text_transforms':
        return TextTransformsDataset(dataset, **kwargs)
    else:
        raise TypeError(f'dataset_preprocessor: {dataset_preprocessor} is not a valid option')
