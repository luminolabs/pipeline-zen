from typing import Any, Dict, Optional

from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchtune import config, modules, utils

from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


# noinspection PyProtocol
class FullFinetuneRecipeSingleDevice(RecipeBase):
    """
    Full fine-tuning recipe for single device training.
    """
    def _setup(self):
        ckpt_dict = self.load_checkpoint(self.cfg.checkpointer)
        self.model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self.tokenizer = config.instantiate(self.cfg.tokenizer)
        self.optimizer = self._setup_optimizer(
            cfg_optimizer=self.cfg.optimizer,
            optimizer_in_bwd=self.cfg.optimizer_in_bwd,
        )
        self.loss_fn = config.instantiate(self.cfg.loss)
        self.sampler, self.dataloader = self.setup_data(
            cfg_dataset=self.cfg.dataset,
            shuffle=self.cfg.shuffle,
            batch_size=self.cfg.batch_size,
        )
        self.steps_per_epoch = (
                len(self.dataloader) // self.gradient_accumulation_steps
        )
        self.global_step = self.epochs_run * self.steps_per_epoch

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg_model)
        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        model.load_state_dict(model_state_dict)
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self.dtype)
        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optimizations for every parameter.
            # noinspection PyTypeChecker
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in self.model.parameters()
            }
            # Register optimizer step hooks on the model to run optimizer in backward.
            utils.register_optim_in_bwd_hooks(model=self.model, optim_dict=optim_dict)
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self.optim_ckpt_wrapper = utils.create_optim_in_bwd_wrapper(
                model=self.model, optim_dict=optim_dict
            )
            # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # use optimizer in backward.
            if opt_state_dict is not None:
                try:
                    self.optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            return None
        else:
            # noinspection PyTypeChecker
            optimizer = config.instantiate(cfg_optimizer, self.model.parameters())
            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)
            return optimizer

    def _save_checkpoint(self):
        checkpoint_dict = {utils.MODEL_KEY: self.model.state_dict()}
        self.checkpointer.save_checkpoint(
            checkpoint_dict,
            epoch=self.epochs_run,
            intermediate_checkpoint=(self.epochs_run < self.total_epochs),
        )


def recipe_main(job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset):
    # Run the recipe
    run_recipe(FullFinetuneRecipeSingleDevice, job_id, user_id, cfg, dataset)
