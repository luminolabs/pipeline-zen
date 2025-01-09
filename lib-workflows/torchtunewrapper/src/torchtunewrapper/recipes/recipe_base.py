import os
from abc import abstractmethod
from logging import Logger
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchtune import training
from torchtune.recipe_interfaces import FTRecipeInterface

from common.agent.job_logger import TorchtunewrapperLoggerAgent
from common.config_manager import config
from common.heartbeats import heartbeat_wrapper
from common.utils import get_artifacts


# noinspection PyProtocol
class RecipeBase(FTRecipeInterface):
    def __init__(self,
                 job_id: str, user_id: str,
                 cfg: DictConfig, dataset: Dataset,
                 logger: Logger, job_logger: TorchtunewrapperLoggerAgent):
        # Set class attributes
        self.job_id = job_id
        self.user_id = user_id
        self.cfg = cfg
        self.logger = logger
        self.job_logger = job_logger
        self.dataset = dataset
        self._tokenizer = None  # Set in the _setup method in child classes

        # Update the application config so that they can be accessed within the thread
        config.set('job_id', job_id)
        config.set('user_id', user_id)

        # Get the number of GPUs available and the rank of the current process
        self.size, self.rank = training.get_world_size_and_rank()

        # Set the PyTorch CUDA allocation configuration
        # This is useful for memory management on GPUs and can be used to prevent OOM errors
        pytorch_cuda_alloc_conf = cfg.get('pytorch_cuda_alloc_conf', None)
        if pytorch_cuda_alloc_conf:
            self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to: {pytorch_cuda_alloc_conf} for GPU #{self.rank}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_cuda_alloc_conf

        self._init(cfg)

    @heartbeat_wrapper('torchtunewrapper', 'load_checkpoint')
    def load_checkpoint(self, cfg_checkpointer: DictConfig):
        self._load_checkpoint(self.cfg.checkpointer)

    @heartbeat_wrapper('torchtunewrapper', 'setup')
    def setup(self) -> None:
        # Log job information
        self._log_job_info()
        # Run the setup method
        self._setup(self.cfg)
        # Set the tokenizer, now that it's been initialized, to the dataset
        self.dataset._model_transform = self._tokenizer

    @heartbeat_wrapper('torchtunewrapper', 'train')
    def train(self) -> None:
        self._train()

    @heartbeat_wrapper('torchtunewrapper', 'save_checkpoint')
    def save_checkpoint(self, epoch: int) -> None:
        self._save_checkpoint(epoch)

    @heartbeat_wrapper('torchtunewrapper', 'cleanup')
    def cleanup(self):
        # Log and send weights and other results to listeners
        self._log_artifacts()
        # Run the cleanup method
        self._cleanup()

    def _log_job_info(self):
        self.job_logger.log_system_specs()
        self.job_logger.log_job_config(OmegaConf.to_container(self.cfg, resolve=True))

    def _log_artifacts(self):
        weight_files, other_files = get_artifacts(self.job_id, self.user_id)
        self.job_logger.log_weights(weight_files, other_files)

    @abstractmethod
    def _init(self, cfg: DictConfig) -> None:
        pass

    @abstractmethod
    def _load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _setup(self, cfg: DictConfig) -> None:
        pass

    @abstractmethod
    def _train(self) -> None:
        pass

    @abstractmethod
    def _save_checkpoint(self, epoch: int) -> None:
        pass

    @abstractmethod
    def _cleanup(self) -> None:
        pass
