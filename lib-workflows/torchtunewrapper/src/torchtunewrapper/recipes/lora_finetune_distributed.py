import os
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.utils import setup_logger
from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


# noinspection PyProtocol
class LoRAFinetuneRecipeDistributed(RecipeBase):
    """
    The LoRA Fine-tuning Recipe for distributed training.
    """
    def __init__(self, *args, **kwargs):
        self.is_lora = True
        super().__init__(*args, **kwargs)

    def setup(self):
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=self.cfg.checkpointer)
        self.model = self.setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
        )
        self.tokenizer = config.instantiate(self.cfg.tokenizer)
        self.optimizer = self.setup_optimizer(
            cfg_optimizer=self.cfg.optimizer,
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
        self.lr_scheduler = self.setup_lr_scheduler(
            cfg_lr_scheduler=self.cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self.steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def setup_model(
            self,
            cfg_model: DictConfig,
            enable_activation_checkpointing: bool,
            base_model_state_dict: Dict[str, Any],
            lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we load the model on CPU with the right
              dtype. To ensure that we don't instantiate ``world_size`` number of models,
              we initialize on meta_device for all ranks other than rank 0.
           b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
              model weights from checkpoint.
           c. While wrapping the model with FSDP, we set ``sync_module_states``
              to TRUE and broadcast module params and buffers from rank 0.
           d. The ``device_id`` param ensures that the FSDP initialization happens on
              the correct device.
        """
        self.lora_rank = cfg_model.lora_rank
        self.lora_alpha = cfg_model.lora_alpha
        self.lora_attn_modules = list(cfg_model.lora_attn_modules)
        self.apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self.apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        if self.is_rank_zero:
            with utils.set_default_dtype(self.dtype):
                model = config.instantiate(cfg_model)

            # The model contains LoRA params which won't have any matching keys in
            # the state dict. As a result, we need to load with strict=False.
            # Before loading the state dict, ensure the state dict keys for the base
            # model and adapters (if available) match the keys in the full LoRA model
            # This is a good sanity check to prevent silent errors
            # noinspection PyTypeChecker
            validate_state_dict_for_lora(
                lora_attn_modules=cfg_model.lora_attn_modules,
                apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
                apply_lora_to_output=getattr(cfg_model, "apply_lora_to_output", False),
                full_model_state_dict_keys=model.state_dict().keys(),
                lora_state_dict_keys=(
                    lora_weights_state_dict.keys()
                    if lora_weights_state_dict is not None
                    else None
                ),
                base_model_state_dict_keys=base_model_state_dict.keys(),
            )

            # Load both the base model weights and (if available) the adapter weights. Both
            # of this should happen only on Rank 0
            model.load_state_dict(base_model_state_dict, strict=False)
            if lora_weights_state_dict:
                model.load_state_dict(lora_weights_state_dict, strict=False)
        else:
            # For non-zero ranks, load the model on meta device
            with utils.set_default_dtype(self.dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self.dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        self.lora_rank = cfg_model.lora_rank
        self.lora_alpha = cfg_model.lora_alpha

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        model = FSDP(
            module=model,
            auto_wrap_policy=utils.lora_fsdp_wrap_policy(
                modules_to_wrap={modules.TransformerDecoderLayer}
            ),
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            device_id=self.device,
            # This recipe does not currently support mixed precision training
            mixed_precision=None,
            # Ensure we broadcast params and buffers from rank 0
            sync_module_states=True,
            # Initialize empty modules on all non-zero ranks
            param_init_fn=(
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                )
                if not self.is_rank_zero
                else None
            ),
        )

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        if self.is_rank_zero:
            memory_stats = utils.get_memory_stats(device=self.device)
            utils.log_memory_stats(memory_stats)

        # Synchronize before training begins
        torch.distributed.barrier()

        return model

    def setup_optimizer(
            self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        # noinspection PyTypeChecker
        optimizer = config.instantiate(cfg_optimizer, self.model.parameters())
        if opt_state_dict:
            opt_state_dict = FSDP.optim_state_dict_to_load(
                self.model, optimizer, opt_state_dict
            )
            optimizer.load_state_dict(opt_state_dict)
        return optimizer

    def setup_lr_scheduler(
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

    def save_checkpoint(self):
        checkpoint_dict = {}
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state_dict = self.model.state_dict()

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self.is_rank_zero:
            # Filter out the adapter keys and weights from the model state dict. These will
            # be saved separately
            adapter_key_filter = lambda x: x in self.adapter_params
            adapter_state_dict = {
                k: v for k, v in cpu_state_dict.items() if adapter_key_filter(k)
            }
            checkpoint_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

            # Merge the adapter weights and base weights to create the model checkpoint
            merged_state_dict = get_merged_lora_ckpt(
                cpu_state_dict,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
            )
            checkpoint_dict.update({utils.MODEL_KEY: merged_state_dict})

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
            checkpoint_dict.update({utils.ADAPTER_CONFIG: adapter_config})

            self.checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=self.epochs_run,
            )

    def cleanup(self):
        destroy_process_group()


def recipe_main(cfg: DictConfig, dataset: Dataset, job_id: str, user_id: str):
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    # Run the recipe
    run_recipe(LoRAFinetuneRecipeDistributed, job_id, user_id, cfg, dataset)
