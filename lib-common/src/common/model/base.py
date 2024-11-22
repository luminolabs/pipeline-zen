import os
from abc import abstractmethod
from logging import Logger

from common.config_manager import config


class BaseModelProvider:
    def __init__(self, url: str, logger: Logger):
        """
        Initialize the BaseDataset.

        :param url: The model uniform resource location
        :param logger: The logger instance
        """
        self.url = url  # ex. hf://meta-llama/Meta-Llama-3.1-70B-Instruct
        self.provider = self.url[:2]  # ex. hf
        self.model_name = self.url[5:]  # ex. meta-llama/Meta-Llama-3.1-70B-Instruct
        self.logger = logger
        self.local_path = self.get_model_cache_dir()

    def get_model_cache_dir(self):
        """
        Where to store the base model locally
        ex: `.cache/models/huggingface/llama3-1-8b`

        :return: The path to the base model cache directory
        """
        cache_dir = os.path.join(config.root_path, config.cache_dir, 'models', self.provider, self.model_name.lower())
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def __call__(self, *args, **kwargs):
        """
        Allows the model provider to be called as a function.

        ex: model_provider = HuggingFaceProvider(url, logger)()
        """
        return self.fetch(**kwargs)

    @abstractmethod
    def fetch(self, **kwargs) -> str:
        """
        Downloads the model.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


def model_provider_factory(url: str, logger: Logger) -> BaseModelProvider:
    """
    Factory function to create a model provider.

    :param url: The base model URL
    :param logger: The logger instance
    :return: The model provider function
    """
    if url.startswith('hf://'):
        from common.model.hugging_face import HuggingFaceProvider
        return HuggingFaceProvider(url, logger)
    else:
        raise ValueError(f"Unknown model provider: {url}")
