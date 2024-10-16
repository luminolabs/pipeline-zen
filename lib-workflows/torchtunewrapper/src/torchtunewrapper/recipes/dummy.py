from typing import Tuple, Dict, Any, List

from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler, DataLoader
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer

from common.comms import heartbeat_wrapper
from common.utils import get_work_dir
from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


class DummyTokenizer(ModelTokenizer):
    def tokenize_messages(
            self, messages: List[Message], **kwargs
    ) -> Tuple[List[int], List[bool]]:
        # Simulate tokenizing messages
        r = [i for i in range(len(messages))], [True for _ in range(len(messages))]
        return r


class Dummy(RecipeBase):
    def setup_tokenizer(self) -> None:
        # Simulate setting up tokenizer
        self.logger.info("Setting up tokenizer")
        
        self.tokenizer = DummyTokenizer()
        self.dataset._tokenizer = self.tokenizer
        self.logger.info("Tokenizer setup complete")
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

    def _save_checkpoint(self):
        # Write some dummy files to simulate saving model weights
        self.logger.info("Saving checkpoint")
        
        work_dir = get_work_dir(self.job_id, self.user_id)
        weights_files = [f'{work_dir}/weights_{i}_3.pt' for i in range(4)]
        other_files = [f'{work_dir}/config.json']
        for file in weights_files + other_files:
            with open(file, 'w') as f:
                f.write('dummy weights')
        self.logger.info("Checkpoint saved")
        return


def recipe_main(job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset):
    # Run the recipe
    run_recipe(Dummy, job_id, user_id, cfg, dataset)
