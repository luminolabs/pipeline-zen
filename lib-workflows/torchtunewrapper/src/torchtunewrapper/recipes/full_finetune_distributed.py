from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import (
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchtune import config, modules, utils
from torchtune.utils.activations import apply_selective_activation_checkpointing

from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe


# noinspection PyProtocol
class FullFinetuneRecipeDistributed(RecipeBase):
    """
    A full fine-tuning recipe for distributed training.
    """
    def setup(self):
        ckpt_dict = self.load_checkpoint(self.cfg.checkpointer)
        self.model = self.setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            memory_efficient_fsdp_wrap=self.cfg.get("memory_efficient_fsdp_wrap", False),
            fsdp_cpu_offload=self.cfg.get("fsdp_cpu_offload", False),
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            ac_mode=self.cfg.get("ac_mode", None),
            ac_option=self.cfg.get("ac_option", None),
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

    def setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        memory_efficient_fsdp_wrap: bool,
        fsdp_cpu_offload: bool,
        model_state_dict: Dict[str, Any],
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
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
        if self.is_rank_zero:
            with utils.set_default_dtype(self.dtype):
                model = config.instantiate(cfg_model)
            model.load_state_dict(model_state_dict)
        else:
            with utils.set_default_dtype(self.dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self.dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        ac_mode = ac_mode
        ac_option = ac_option

        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(model, ac_mode, ac_option)

        # Wrap the model with FSDP. This will ensure that the model is sharded
        # across all available GPUs.
        model = FSDP(
            module=model,
            auto_wrap_policy=utils.get_full_finetune_fsdp_wrap_policy(
                memory_efficient_fsdp_wrap=memory_efficient_fsdp_wrap,
                modules_to_wrap={modules.TransformerDecoderLayer},
            ),
            cpu_offload=CPUOffload(offload_params=fsdp_cpu_offload),
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
                ) if not self.is_rank_zero else None
            ),
        )

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        # Original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
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
            checkpoint_dict.update({utils.MODEL_KEY: cpu_state_dict})
            self.checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=self.epochs_run,
                intermediate_checkpoint=(self.epochs_run < self.total_epochs),
            )

    def cleanup(self):
        destroy_process_group()


def recipe_main(cfg: DictConfig, dataset: Dataset, job_id: str, user_id: str):
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        utils.set_torch_num_threads()

    # Run the recipe
    run_recipe(FullFinetuneRecipeDistributed, job_id, user_id, cfg, dataset)
