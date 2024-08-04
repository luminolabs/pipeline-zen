import os
from logging import Logger
from typing import Dict

from google.cloud import storage

from common.dataset.provider.base import BaseDatasetProvider


class GcpBucket(BaseDatasetProvider):
    """
    GCP Bucket Dataset Provider

    NOTE: This is more of a downloader than a dataset provider.
    LATER: To use as a dataset provider, `_getitem()` and `_len()` methods need to be implemented.
    """

    def fetch(self, logger: Logger, **kwargs) -> None:
        """
        Fetches the dataset from a Google Cloud Storage bucket.

        Args:
            logger: The logger instance
            **kwargs: Additional keyword arguments
        Raises:
            ValueError: If the bucket name is not `lum-pipeline-zen-jobs-us`
        """
        # Parse the GCS URL to get the bucket name and source blob name
        gs_url = self.dataset_id  # ex. gs://bucket-name/path-to-dataset/dataset-name
        bucket_name = gs_url.split('/')[2]  # ex. bucket-name
        source_blob_name = '/'.join(gs_url.split('/')[3:])  # ex. path-to-dataset/dataset-name
        # Generate the destination file name, ex: .cache/gcp_bucket/path-to-dataset/dataset-name
        destination_file_name = os.path.join(self.get_dataset_cache_dir(), *source_blob_name.split('/'))

        # Raise an error if the bucket name is not allowed
        if bucket_name != 'lum-pipeline-zen-jobs-us':
            raise ValueError(f'Bucket name `{bucket_name}` is not allowed; use `lum-pipeline-zen-jobs-us` instead')

        # Download the dataset from GCS
        download_from_gcs(bucket_name, source_blob_name, destination_file_name)
        # Set the dataset to the destination file name; later `_getitem()` and `_len()` methods can use this
        self.dataset = destination_file_name

    def _getitem(self, item: int) -> Dict:
        # LATER: We don't need this method for now
        raise NotImplementedError()

    def _len(self) -> int:
        # LATER: We don't need this method for now
        raise NotImplementedError()


def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """
    Downloads a file from a Google Cloud Storage bucket.

    Args:
        bucket_name: The name of the bucket
        source_blob_name: The name of the source blob
        destination_file_name: The name of the destination file
    """
    # Initialize a client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Get the blob
    blob = bucket.blob(source_blob_name)
    # Download the blob to a local file
    blob.download_to_filename(destination_file_name)
