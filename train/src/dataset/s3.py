import boto3
from torch.utils.data import Dataset

from dataset.base_dataset import BaseDataset


class S3(BaseDataset):
    def __init__(self, bucket_name: str, object_key: str,
                 local_path: str) -> None:
        self.bucket_name = bucket_name
        self.object_key = object_key
        self.local_path = local_path

    async def load(self) -> None:
        # TODO: is there support for async download?
        s3 = boto3.client('s3')
        s3.download_file(Bucket=self.bucket_name, Key=self.object_key,
                         Filename=self.local_path)

    async def to_torch_dataset(self, split: str) -> Dataset:
        # TODO
        pass
