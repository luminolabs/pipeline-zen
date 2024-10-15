from abc import abstractmethod
from typing import Tuple, Dict, Any

from omegaconf import DictConfig
from torch.utils.data import DistributedSampler, DataLoader, Dataset

from common.comms import heartbeat_wrapper
from common.utils import get_work_dir
from torchtunewrapper.recipes.recipe_base import RecipeBase


class Dummy(RecipeBase):
    def setup_tokenizer(self) -> None:
        return

    def setup_data(
            self,
            shuffle: bool,
            batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        # Simulate setting up data
        self.logger.info("Setting up data")
        sampler = DistributedSampler(self.dataset)
        loader = DataLoader(self.dataset, shuffle=shuffle, sampler=sampler, batch_size=batch_size)
        self.logger.info("Data setup complete")
        return sampler, loader

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        # Simulate loading checkpoint
        self.logger.info("Loading checkpoint")
        r = {'foo': 'bar'}
        self.logger.info("Checkpoint loaded")
        return r

    @heartbeat_wrapper('torchtunewrapper', 'train')
    def train(self) -> None:
        # Simulate training
        self.logger.info("Started training")
        self.logger.info("Training complete")
        return

    def cleanup(self):
        # Simulate cleanup
        self.logger.info("Cleaning up")
        self.logger.info("Cleanup complete")
        pass

    def _setup(self):
        # Simulate setup
        self.logger.info("Setting up")
        self.logger.info("Setup complete")
        pass

    @abstractmethod
    def _save_checkpoint(self):
        # Write some dummy files to simulate saving model weights
        self.logger.info("Saving checkpoint")
        work_dir = get_work_dir('-1', '-1')
        weights_files = [f'{work_dir}/weights_{i}_3.pth' for i in range(4)]
        for file in weights_files:
            with open(file, 'w') as f:
                f.write('dummy weights')
        self.logger.info("Checkpoint saved")
        return
