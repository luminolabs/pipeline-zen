import os
import time

from functools import partial
from logging import Logger
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from torch import nn
from torch.distributed import init_process_group
from torch.distributed.fsdp import (
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from torchtune import config, modules, utils
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils.activations import apply_selective_activation_checkpointing

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.utils import setup_logger


# noinspection PyProtocol
class FullFinetuneRecipeDistributed(FTRecipeInterface):
    """
    A full fine-tuning recipe for distributed training.
    """
    def __init__(self, cfg: DictConfig,
                 logger: Logger, scores_agent: TorchtunewrapperScoresAgent, dataset: Dataset):
        self.cfg = cfg
        self.logger = logger
        self.scores_agent = scores_agent
        self.dataset = dataset

        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError("Full fp16 training is not supported with this recipe. "
                             "Please use bf16 or fp32 instead.")

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError("Using fused optimizer on CPU is only supported in PyTorch nightly.")

        # Get the number of GPUs and GPU index for this process
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training configuration
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # Set the PyTorch CUDA allocation configuration
        # This is useful for memory management on GPUs and can be used to prevent OOM errors
        pytorch_cuda_alloc_conf = cfg.get("pytorch_cuda_alloc_conf", None)
        if pytorch_cuda_alloc_conf:
            self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to: {pytorch_cuda_alloc_conf} for GPU #{rank}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_cuda_alloc_conf

        # Set the seed for reproducibility
        self.seed = utils.set_seed(seed=cfg.seed)

        # Initialize the recipe state
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.global_step = 0

        # Initialize various variables used in this recipe
        self._steps_per_epoch = None
        self._dataloader = None
        self._sampler = None
        self._loss_fn = None
        self._optimizer = None
        self._tokenizer = None
        self._model = None
        self._checkpointer = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self):
        ckpt_dict = self.load_checkpoint(self.cfg.checkpointer)
        self._model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            memory_efficient_fsdp_wrap=self.cfg.get("memory_efficient_fsdp_wrap", False),
            fsdp_cpu_offload=self.cfg.get("fsdp_cpu_offload", False),
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            ac_mode=self.cfg.get("ac_mode", None),
            ac_option=self.cfg.get("ac_option", None),
        )
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=self.cfg.optimizer,
        )
        self._loss_fn = config.instantiate(self.cfg.loss)
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=self.cfg.dataset,
            shuffle=self.cfg.shuffle,
            batch_size=self.cfg.batch_size,
        )
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        self.global_step = self.epochs_run * self._steps_per_epoch

    def _setup_model(
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
        if self._is_rank_zero:
            with utils.set_default_dtype(self._dtype):
                model = config.instantiate(cfg_model)
            model.load_state_dict(model_state_dict)
        else:
            with utils.set_default_dtype(self._dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self._dtype == torch.bfloat16:
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
            device_id=self._device,
            # This recipe does not currently support mixed precision training
            mixed_precision=None,
            # Ensure we broadcast params and buffers from rank 0
            sync_module_states=True,
            # Initialize empty modules on all non-zero ranks
            param_init_fn=(
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                ) if not self._is_rank_zero else None
            ),
        )

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        # Original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        if self._is_rank_zero:
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        # Synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            opt_state_dict = FSDP.optim_state_dict_to_load(
                self._model, optimizer, opt_state_dict
            )
            optimizer.load_state_dict(opt_state_dict)
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        world_size, rank = utils.get_world_size_and_rank()
        ds = self.dataset
        ds._tokenizer = self._tokenizer
        packed = cfg_dataset.get("packed", False)
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            )
            if not packed
            else None,
        )
        return sampler, dataloader

    def save_checkpoint(self):
        checkpoint_dict = {}
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state_dict = self._model.state_dict()
        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:
            checkpoint_dict.update({utils.MODEL_KEY: cpu_state_dict})
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=self.epochs_run,
                intermediate_checkpoint=(self.epochs_run < self.total_epochs),
            )

    def train(self):
        utils.cleanup_before_training()
        _, rank = utils.get_world_size_and_rank()
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        running_loss = 0
        num_tokens = 0
        t_step_start = time.perf_counter()

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            # Log per-epoch timestamps
            t_epoch_start = time.perf_counter()

            for idx, batch in enumerate(self._dataloader):
                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)
                # Free logits otherwise it peaks backward memory
                del logits

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Log per-step metrics and timestamps
                    time_per_step = time.perf_counter() - t_step_start
                    mem_stats = utils.get_memory_stats(device=self._device)
                    self.scores_agent.log_step(
                        gpu_rank=rank,
                        step_num=self.global_step,
                        step_len=self._steps_per_epoch,
                        step_loss=running_loss.item(),
                        step_lr=self._optimizer.param_groups[0]["lr"],
                        step_tokens_per_second=num_tokens / time_per_step,
                        step_tokens=num_tokens,
                        step_peak_memory_active=mem_stats.get("peak_memory_active"),
                        step_peak_memory_alloc=mem_stats.get("peak_memory_alloc"),
                        step_peak_memory_reserved=mem_stats.get("peak_memory_reserved"),
                        step_time_elapsed_s=time_per_step,
                        epoch_num=curr_epoch + 1,
                        epoch_len=self.total_epochs,)

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t_step_start = time.perf_counter()

            # Log per-epoch timestamps
            time_per_epoch = time.perf_counter() - t_epoch_start
            self.scores_agent.log_epoch(gpu_rank=rank, epoch_num=curr_epoch + 1, epoch_len=self.total_epochs,
                                        epoch_time_elapsed_s=time_per_epoch)
            self.epochs_run += 1

    def cleanup(self):
        torch.distributed.destroy_process_group()


def recipe_main(cfg: DictConfig, dataset: Dataset, job_id: str, user_id: str):
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        utils.set_torch_num_threads()

    # Set up the main logger
    logger = setup_logger("torchtunewrapper_recipe", job_id, user_id)
    # A logger for logging scores; also propagates to main logger
    scores_logger = setup_logger('torchtunewrapper_recipe.metrics', job_id, user_id, add_stdout=False)
    # Setup logging and bigquery agent for scores
    scores_agent = TorchtunewrapperScoresAgent(job_id, scores_logger)
    # Initialize the recipe and start training
    recipe = FullFinetuneRecipeDistributed(cfg, logger, scores_agent, dataset)
    recipe.setup()
    recipe.train()
    recipe.save_checkpoint()
    recipe.cleanup()
