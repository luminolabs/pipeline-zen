from pathlib import Path
import shutil
from common.dataset.base import BaseDatasetProvider


class FileSystemProvider(BaseDatasetProvider):
    """
    File system Dataset Provider
    """

    def fetch(self, **kwargs) -> str:
        """
        Fetches the dataset from a file in the local file system.

        :param kwargs: Additional keyword arguments
        """
        # Raise an error if the url prefix is not what we expect
        if not self.url.startswith('file://'):
            raise ValueError(f'Filesystem datasets should start with `file://` only')
        # Parse the Filesystem URL to remove prefix.
        src = Path(self.url.replace('file://', ''))
        dst = Path(self.local_path)
        self.logger.info(f"Copying FileSystem dataset: {src} -> {dst}")
        shutil.copy(src, dst)
        return self.local_path
