import time

import huggingface_hub

from common.config_manager import config
from common.model.base import BaseModelProvider


class HuggingFaceProvider(BaseModelProvider):
    """
    Hugging Face model provider.
    """

    def fetch(self, **kwargs) -> str:
        """
        Fetch the model from the Hugging Face Hub.

        :param kwargs: Additional keyword arguments
        :return: The local path to the model
        """
        self.logger.info(f"Downloading Hugging Face model: {self.url}")
        # Add the Hugging Face token to the kwargs
        kwargs['token'] = config.huggingface_token
        # Set the model download location to the local path
        kwargs['cache_dir'] = self.local_path
        # Download the model from the Hugging Face Hub and return the local path

        # retry 5 times
        for i in range(5):
            try:
                path = huggingface_hub.snapshot_download(self.model_name, **kwargs)
                return path
            except Exception as e:
                self.logger.error(f"Error downloading model: {e}")
                self.logger.info(f"Retrying in 10 seconds...")
                time.sleep(10)
