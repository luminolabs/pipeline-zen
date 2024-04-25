from common.dataset.kind.single_label_dataset import SingleLabelDataset
from common.dataset.provider.base import BaseDatasetProvider


def dataset_kind_factory(dataset_kind: str,
                         dataset: BaseDatasetProvider,
                         input_col: str,
                         **kwargs):
    """
    Factory function for instantiating a dataset kind.

    :param dataset_kind: Short name of the dataset kind
    :param dataset: Dataset to iterate on; must be a `BaseDatasetProvider`
    :param input_col: The name of the input column in the dataset
    :param kwargs: Additional keyword arguments to pass to the dataset kind
    :return: A dataset kind
    """
    if dataset_kind == 'single_label':
        return SingleLabelDataset(dataset=dataset, input_col=input_col, **kwargs)
    elif dataset_kind == '...':
        pass
    else:
        raise TypeError(f'dataset_kind: {dataset_kind} is not a valid option')
