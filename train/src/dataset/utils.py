from dataset.base_dataset import BaseDataset
from dataset.huggingface import HuggingFace
from dataset.s3 import S3


def dataset_factory(strategy: str, **kwargs) -> BaseDataset:
    """
    Since the dataset will be defined in the job configuration, we need a way to go from
    configuration, which is a string, to class.

    For example:
      job_config.yaml
        - dataset_store: huggingface
        - dataset_store_args: arg1,arg2

    The above will be fed to this factory, so that the code that pulls the dataset
    stays unaware of the provider.

    So, the code that pulls the dataset will run something like this:

    ```
    dataset = factory(strategy=dataset_store, *dataset_store_args)
    torch_dataset = dataset.to_torch_dataset(split='train')
    ```

    """
    if strategy == 'huggingface':
        return HuggingFace(dataset_path=kwargs.get('dataset_path'))
    if strategy == 's3':
        return S3(bucket_name=kwargs.get('bucket_name'),
                  object_key=kwargs.get('object_key'))