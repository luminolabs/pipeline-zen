from common.dataset.base import BaseDatasetProvider


class FileSystemProvider(BaseDatasetProvider):
    """
    File system Dataset Provider
    """

    def fetch(self, **kwargs) -> None:
        """
        Fetches the dataset from a file in the local file system.

        :param kwargs: Additional keyword arguments
        """
        return self.local_path
