import os
from abc import abstractmethod
from logging import Logger

from common.utils import get_work_dir


class BaseDatasetProvider:
    def __init__(self, url: str,
                 job_id: str, user_id: str,
                 logger: Logger):
        """
        Initialize the BaseDataset.

        :param url: The dataset uniform resource location
        :param job_id: The job ID
        :param user_id: The user ID
        :param logger: The logger instance
        """
        self.url = url
        self.job_id = job_id
        self.user_id = user_id
        self.logger = logger
        self.local_path = str(os.path.join(get_work_dir(job_id, user_id), 'dataset.jsonl'))

    def __call__(self, *args, **kwargs):
        """
        Allows the dataset provider to be called as a function.

        ex: dataset_provider = GcpBucketProvider(url, job_id, user_id, logger)()
        """
        return self.fetch(**kwargs)

    @abstractmethod
    def fetch(self, **kwargs) -> str:
        """
        Downloads the dataset.

        :param kwargs: Additional keyword arguments
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


def dataset_provider_factory(url: str,
                             job_id: str, user_id: str,
                             logger: Logger) -> BaseDatasetProvider:
    """
    Factory function to create a dataset provider.
    """
    if url.startswith('gs://'):
        from common.dataset.gcp_bucket import GcpBucketProvider
        return GcpBucketProvider(url, job_id, user_id, logger)
    elif url.startswith('file://'):
        from common.dataset.file_system import FileSystemProvider
        return FileSystemProvider(url, job_id, user_id, logger)
    else:
        raise ValueError(f'Unknown dataset provider: {url}')
