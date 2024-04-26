from common.dataset.provider.huggingface import HuggingFace


def dataset_provider_factory(dataset_provider: str, dataset_id: str, split: str):
    """
    Factory method for creating a dataset provider instance

    :param dataset_provider:
    :param dataset_id:
    :param split:
    :return:
    """
    print(f'Using `{dataset_id}.{split}` from `{dataset_provider}`')
    if dataset_provider == 'huggingface':
        return HuggingFace(dataset_id, split)
    if dataset_provider == '...':
        pass
    else:
        raise TypeError(f'dataset_provider: {dataset_provider} is not a valid option')
