import os
import time
from functools import partial
from logging import Logger
from typing import Tuple, Dict, Any, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler, DataLoader, Dataset
from torchtune import utils, config
from torchtune.recipe_interfaces import FTRecipeInterface

from common.agents.model_scores import TorchtunewrapperScoresAgent


# noinspection PyProtocol
class RecipeBase(FTRecipeInterface):
    def __init__(self, cfg: DictConfig,
                 logger: Logger, scores_agent: TorchtunewrapperScoresAgent, dataset: Optional[Dataset] = None):
        self.cfg = cfg
        self.logger = logger
        self.scores_agent = scores_agent
        self.dataset = dataset

        self.device = utils.get_device(device=cfg.device)
        self.dtype = utils.get_dtype(cfg.dtype, device=self.device)

        if self.dtype == torch.float16:
            raise ValueError(
                "Full fp16 training is not supported with this recipe. "
                "Please use bf16 or fp32 instead."
            )

        if (
                self.dtype == torch.bfloat16
                and self.device != torch.device("cpu")
                and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("Full bf16 training is not supported on this hardware.")

        if (
                cfg.fsdp_cpu_offload
                and cfg.optimizer.fused
                and not utils.torch_version_ge("2.4.0")
        ):
            raise ValueError("Using fused optimizer on CPU is only supported in PyTorch nightly.")

        if cfg.gradient_accumulation_steps > 1 and cfg.optimizer_in_bwd:
            raise ValueError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

        _, rank = utils.get_world_size_and_rank()
        self.is_rank_zero = rank == 0

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
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.steps_per_epoch = None
        self.optim_ckpt_wrapper = None

        # Initialize objects variables
        self.dataloader = None
        self.sampler = None
        self.loss_fn = None
        self.optimizer = None
        self.tokenizer = None
        self.model = None
        self.checkpointer = None
        self.lr_scheduler = None
        # LoRA-specific attributes
        self.is_lora = False
        self.apply_lora_to_output = None
        self.apply_lora_to_mlp = None
        self.lora_attn_modules = None
        self.adapter_params = None
        self.lora_alpha = None
        self.lora_rank = None

    def setup_data(
            self,
            cfg_dataset: DictConfig,
            shuffle: bool,
            batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        world_size, rank = utils.get_world_size_and_rank()
        ds = self.dataset
        ds.tokenizer = self.tokenizer
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
                padding_idx=self.tokenizer.pad_id,
                ignore_idx=self.loss_fn.ignore_index,
            )
            if not packed
            else None,
        )
        return sampler, dataloader

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self.checkpointer = config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self.checkpointer.load_checkpoint()
        return checkpoint_dict

    def train(self) -> None:
        utils.cleanup_before_training()
        _, rank = utils.get_world_size_and_rank()
        if not self.cfg.optimizer_in_bwd:
            self.optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        running_loss = 0
        num_tokens = 0
        t_step_start = time.perf_counter()

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self.sampler.set_epoch(curr_epoch)

            # Log per-epoch timestamps
            t_epoch_start = time.perf_counter()

            for idx, batch in enumerate(self.dataloader):
                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self.device)
                num_tokens += tokens.numel()
                labels = labels.to(self.device)
                mask = mask.to(self.device) if mask is not None else None
                input_pos = (
                    input_pos.to(self.device) if input_pos is not None else None
                )

                logits = self.model(tokens, mask=mask, input_pos=input_pos)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self.loss_fn(logits, labels)
                # Free logits otherwise it peaks backward memory
                del logits

                loss = loss / self.gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    if not self.cfg.optimizer_in_bwd:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.is_lora:
                            self.lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Log per-step metrics and timestamps
                    time_per_step = time.perf_counter() - t_step_start
                    mem_stats = utils.get_memory_stats(device=self.device)
                    self.scores_agent.log_step(
                        gpu_rank=rank,
                        step_num=self.global_step,
                        step_len=self.steps_per_epoch,
                        step_loss=running_loss.item(),
                        step_lr=(
                            self.optim_ckpt_wrapper.get_optim_key("lr")
                            if self.optim_ckpt_wrapper and self.cfg.optimizer_in_bwd
                            else self.optimizer.param_groups[0]["lr"]
                        ),
                        step_tokens_per_second=num_tokens / time_per_step,
                        step_tokens=num_tokens,
                        step_peak_memory_active=mem_stats.get("peak_memory_active"),
                        step_peak_memory_alloc=mem_stats.get("peak_memory_alloc"),
                        step_peak_memory_reserved=mem_stats.get("peak_memory_reserved"),
                        step_time_elapsed_s=time_per_step,
                        epoch_num=curr_epoch + 1,
                        epoch_len=self.total_epochs, )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t_step_start = time.perf_counter()

            # Log per-epoch timestamps
            time_per_epoch = time.perf_counter() - t_epoch_start
            self.scores_agent.log_epoch(gpu_rank=rank, epoch_num=curr_epoch + 1,
                                        epoch_len=self.total_epochs,
                                        epoch_time_elapsed_s=time_per_epoch)
            self.epochs_run += 1
