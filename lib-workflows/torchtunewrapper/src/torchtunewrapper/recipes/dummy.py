from typing import Tuple, Dict, Any, List

from datasets import Dataset
from omegaconf import DictConfig
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer

from common.utils import get_work_dir
from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


class DummyTokenizer(ModelTokenizer):
    # noinspection PyProtocol
    def tokenize_messages(
            self, messages: List[Message], **kwargs
    ) -> Tuple[List[int], List[bool]]:
        # Simulate tokenizing messages
        r = [i for i in range(len(messages))], [True for _ in range(len(messages))]
        return r


class Dummy(RecipeBase):

    def _init(self, cfg: DictConfig) -> None:
        # Simulate initialization
        self.logger.info("Initialized")

    def _load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        # Simulate loading checkpoint
        self.logger.info("Loading checkpoint")
        r = {'foo': 'bar'}
        self.logger.info("Checkpoint loaded")
        return r

    def _train(self) -> None:
        # Simulate training
        self.logger.info("Started training")
        self.logger.info("Training complete")
        return

    def _cleanup(self) -> None:
        # Simulate cleanup
        self.logger.info("Cleaning up")
        self.logger.info("Cleanup complete")

    def _setup(self, cfg: DictConfig) -> None:
        # Simulate setup
        self.logger.info("Setting up")
        self.logger.info("Setup complete")

    def _save_checkpoint(self, epoch: int) -> None:
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
