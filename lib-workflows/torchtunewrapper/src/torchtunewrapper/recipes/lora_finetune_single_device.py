from logging import Logger
from typing import Any, Dict, Optional

from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)

from common.agents.model_scores import TorchtunewrapperScoresAgent
from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


# noinspection PyProtocol
class LoRAFinetuneRecipeSingleDevice(RecipeBase):
    """
    Recipe for LoRA fine-tuning on a single device.
    """
    def __init__(self, job_id: str, user_id: str,
                 cfg: DictConfig, dataset: Dataset,
                 logger: Logger, scores_agent: TorchtunewrapperScoresAgent):
        self.is_lora = True
        super().__init__(job_id, user_id, cfg, dataset, logger, scores_agent)

    def _setup(self):
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=self.cfg.checkpointer)
        self.model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.enable_activation_checkpointing,
            model_state_dict=checkpoint_dict[utils.MODEL_KEY],
        )
        self.tokenizer = config.instantiate(self.cfg.tokenizer)
        self.optimizer = self._setup_optimizer(
            cfg_optimizer=self.cfg.optimizer,
        )
        self.loss_fn = config.instantiate(self.cfg.loss)
        self.sampler, self.dataloader = self.setup_data(
            shuffle=self.shuffle,
            batch_size=self.batch_size,
        )
        self.steps_per_epoch = (
            len(self.dataloader) // self.gradient_accumulation_steps
        )
        self.lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=self.lr_scheduler,
            num_training_steps=self.total_epochs * self.steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg_model)

        self.lora_rank = self.lora_rank
        self.lora_alpha = self.lora_alpha
        self.lora_attn_modules = list(self.lora_attn_modules)
        self.apply_lora_to_mlp = self.apply_lora_to_mlp
        self.apply_lora_to_output = self.apply_lora_to_output
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            model_state_dict, strict=False
        )
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self.lora_attn_modules,
            apply_lora_to_mlp=self.apply_lora_to_mlp,
            apply_lora_to_output=self.apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        utils.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self.dtype
        )
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        # noinspection PyTypeChecker
        optimizer = config.instantiate(cfg_optimizer, self.model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        self.logger.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        # noinspection PyTypeChecker
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self.optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        return lr_scheduler

    def _save_checkpoint(self):
        ckpt_dict = {}
        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})
        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self.model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})
        adapter_config = {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": get_lora_module_names(
                self.lora_attn_modules,
                self.apply_lora_to_mlp,
                self.apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({utils.ADAPTER_CONFIG: adapter_config})
        self.checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=self.epochs_run,
        )


def recipe_main(job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset):
    # Run the recipe
    run_recipe(LoRAFinetuneRecipeSingleDevice, job_id, user_id, cfg, dataset)
