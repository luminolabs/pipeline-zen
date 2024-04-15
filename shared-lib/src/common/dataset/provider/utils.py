from common.dataset.provider.base import BaseDatasetProvider
from common.dataset.provider.huggingface import HuggingFace


def dataset_provider_factory(dataset_provider: str, dataset_id: str, split: str) -> BaseDatasetProvider:
    if dataset_provider == 'huggingface':
        return HuggingFace(dataset_id, split)
    if dataset_provider == 's3':
        # TODO
        pass
