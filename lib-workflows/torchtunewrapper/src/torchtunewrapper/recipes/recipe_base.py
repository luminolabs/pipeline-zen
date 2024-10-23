import os
import time
from abc import abstractmethod
from functools import partial
from logging import Logger
from typing import Tuple, Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DistributedSampler, DataLoader, Dataset
from torchtune import utils, config as tt_config

from common.agent.job_logger import TorchtunewrapperLoggerAgent
from common.comms import heartbeat_wrapper
from common.config_manager import config
from common.utils import is_local_env, get_artifacts


# noinspection PyProtocol
class RecipeBase:
    def __init__(self,
                 job_id: str, user_id: str,
                 cfg: DictConfig, dataset: Dataset,
                 logger: Logger, job_logger: TorchtunewrapperLoggerAgent):

        self.job_id = job_id
        self.user_id = user_id
        self.cfg = cfg
        self.logger = logger
        self.job_logger = job_logger
        self.dataset = dataset

        # Update the application config so that they can be accessed within the thread
        config.set('job_id', job_id)
        config.set('user_id', user_id)

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
        self.apply_lora_to_output = cfg.model.get('apply_lora_to_output', False)
        self.apply_lora_to_mlp = cfg.model.get('apply_lora_to_mlp', False)
        self.lora_attn_modules = cfg.model.get('lora_attn_modules', [])
        self.lora_alpha = cfg.model.get('lora_alpha', None)
        self.lora_rank = cfg.model.get('lora_rank', None)
        self.adapter_params = None
        # Other configuration
        self.device = utils.get_device(
            # Use MPS locally since we're all on Apple silicon
            cfg.get('device', 'cuda') if not is_local_env() else "mps")
        self.dtype = utils.get_dtype(
            # Use fp32 locally since we're all on Apple silicon, and bf* is not supported
            cfg.dtype if not is_local_env() else "fp32",
            device=self.device)
        self.gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
        self.fsdp_cpu_offload = cfg.get('fsdp_cpu_offload', False)
        self.memory_efficient_fsdp_wrap = cfg.get('memory_efficient_fsdp_wrap', False)
        self.fused = cfg.optimizer.get('fused', False)
        self.optimizer_in_bwd = cfg.get('optimizer_in_bwd', False)
        self.pytorch_cuda_alloc_conf = cfg.get('pytorch_cuda_alloc_conf', None)
        self.enable_activation_checkpointing = cfg.get('enable_activation_checkpointing', True)
        self.dataset_packed = cfg.dataset.get('packed', False)
        # Distributed environment variables
        _, rank = utils.get_world_size_and_rank()
        self.is_rank_zero = rank == 0
        # State variables
        self.total_epochs = cfg.get('epochs', 1)
        self.shuffle = cfg.get('shuffle', True)
        self.batch_size = cfg.get('batch_size', 2)
        self.optim_ckpt_wrapper = None
        # Training variables
        self.epochs_run = 0
        self.global_step = 0
        self.steps_per_epoch = None
        # Set the seed for reproducibility
        self.seed = utils.set_seed(seed=cfg.get('seed'))

        # Set the PyTorch CUDA allocation configuration
        # This is useful for memory management on GPUs and can be used to prevent OOM errors
        if self.pytorch_cuda_alloc_conf:
            self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to: {self.pytorch_cuda_alloc_conf} for GPU #{rank}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.pytorch_cuda_alloc_conf

        # Validate configuration
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
            self.fsdp_cpu_offload
            and self.fused
            and not utils.torch_version_ge("2.4.0")
        ):
            raise ValueError("Using fused optimizer on CPU is only supported in PyTorch nightly.")
        if self.gradient_accumulation_steps > 1 and self.optimizer_in_bwd:
            raise ValueError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

    def setup_tokenizer(self) -> None:
        self.tokenizer = tt_config.instantiate(self.cfg.tokenizer)
        self.dataset._tokenizer = self.tokenizer

    def setup_data(
            self,
            shuffle: bool,
            batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        world_size, rank = utils.get_world_size_and_rank()
        ds = self.dataset
        ds._tokenizer = self.tokenizer
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
            ) if not self.dataset_packed else None,
        )
        return sampler, dataloader

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self.checkpointer = tt_config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self.checkpointer.load_checkpoint()
        return checkpoint_dict

    @heartbeat_wrapper('torchtunewrapper', 'train')
    def train(self) -> None:
        if self.device.type == "cuda":
            utils.cleanup_before_training()
        _, rank = utils.get_world_size_and_rank()
        if not self.optimizer_in_bwd:
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
            epoch_step = 0

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
                    if not self.optimizer_in_bwd:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.is_lora:
                            self.lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1
                    epoch_step += 1

                    # Log per-step metrics and timestamps
                    time_per_step = time.perf_counter() - t_step_start
                    mem_stats = utils.get_memory_stats(device=self.device)
                    self.job_logger.log_step(
                        gpu_rank=rank,
                        step_num=epoch_step,
                        step_len=self.steps_per_epoch,
                        step_loss=running_loss.item(),
                        step_lr=(
                            self.optim_ckpt_wrapper.get_optim_key("lr")
                            if self.optim_ckpt_wrapper and self.optimizer_in_bwd
                            else self.optimizer.param_groups[0]["lr"]
                        ),
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
            self.job_logger.log_epoch(gpu_rank=rank, epoch_num=curr_epoch + 1,
                                      epoch_len=self.total_epochs,
                                      epoch_time_elapsed_s=time_per_epoch)

            # Update the epoch count
            self.epochs_run += 1

    @heartbeat_wrapper('torchtunewrapper', 'save_weights')
    def save_checkpoint(self) -> None:
        self._save_checkpoint()

    @heartbeat_wrapper('torchtunewrapper', 'setup')
    def setup(self):
        # Log job information
        self._log_job_info()
        # Common setup
        if not self.tokenizer:
            self.setup_tokenizer()
        # Recipe-specific setup
        return self._setup()

    @heartbeat_wrapper('torchtunewrapper', 'cleanup')
    def cleanup(self):
        # Log and send weights and other results to listeners
        self._log_artifacts()
        # Recipe-specific cleanup
        self._cleanup()
        pass

    def _log_job_info(self):
        self.job_logger.log_system_specs()
        self.job_logger.log_job_config(OmegaConf.to_container(self.cfg, resolve=True))

    def _log_artifacts(self):
        weight_files, other_files = get_artifacts(self.job_id, self.user_id)
        self.job_logger.log_weights(weight_files, other_files)

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def _save_checkpoint(self):
        pass

    @abstractmethod
    def _cleanup(self):
        pass
