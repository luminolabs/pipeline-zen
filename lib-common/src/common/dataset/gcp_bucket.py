from common.config_manager import config
from common.dataset.base import BaseDatasetProvider
from common.gcp import download_object


class GcpBucketProvider(BaseDatasetProvider):
    """
    GCP Bucket Dataset Provider
    """

    def fetch(self, **kwargs) -> None:
        """
        Fetches the dataset from a Google Cloud Storage bucket.

        :param kwargs: Additional keyword arguments
        """
        self.logger.info(f"Downloading GCP bucket dataset: {self.url}")
        # Raise an error if the bucket name is not what we expect
        if not self.url.startswith(f'gs://lum-') or self.url.find('-pipeline-zen-datasets') == -1:
            raise ValueError(f'Upload datasets to `gs://lum-*-pipeline-zen-datasets/*` '
                             f'only; got {self.url}')
        # Parse the GCS URL to get the bucket name and source blob name
        gs_url = self.url  # ex. gs://bucket-name/path-to-dataset/dataset-name
        bucket_name = gs_url.split('/')[2]  # ex. lum-dev-pipeline-zen-jobs-us
        source_blob_name = '/'.join(gs_url.split('/')[3:])  # ex. datasets/<user_id>/dataset-name.jsonl
        # Download the dataset from GCS and return the local path
        download_object(bucket_name, source_blob_name, self.local_path)
        return self.local_path
