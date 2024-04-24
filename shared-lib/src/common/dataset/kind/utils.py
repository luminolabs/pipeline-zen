from common.dataset.kind.base import BaseDatasetKind
from common.dataset.kind.single_label_dataset import SingleLabelDataset
from common.dataset.provider.base import BaseDatasetProvider


def dataset_kind_factory(dataset_kind: str, dataset: BaseDatasetProvider,
                         **kwargs) -> BaseDatasetKind:
    """
    Factory function for instantiating a dataset kind.

    :param dataset_kind: Short name of the dataset kind
    :param dataset: Dataset to iterate on; must be a `BaseDatasetProvider`
    :param kwargs: Additional keyword arguments to pass to the dataset kind
    :return: A dataset kind
    """
    if dataset_kind == 'single_label':
        return SingleLabelDataset(dataset, **kwargs)
    if dataset_kind == '...':
        pass
