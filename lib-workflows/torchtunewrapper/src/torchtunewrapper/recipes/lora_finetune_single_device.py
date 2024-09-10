import os
import time

from functools import partial
from logging import Logger
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.utils import setup_logger


# noinspection PyProtocol
class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Recipe for LoRA fine-tuning on a single device.
    """
    def __init__(self, cfg: DictConfig,
                 logger: Logger, scores_agent: TorchtunewrapperScoresAgent, dataset: Optional[Dataset] = None):
        self.cfg = cfg
        self.logger = logger
        self.scores_agent = scores_agent
        self.dataset = dataset
        
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._dtype == torch.float16:
            raise RuntimeError("Full fp16 training is not supported with this recipe. "
                               "Please use bf16 or fp32 instead.")

        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Full bf16 training is not supported on this hardware.")

        # Set the PyTorch CUDA allocation configuration
        # This is useful for memory management on GPUs and can be used to prevent OOM errors
        pytorch_cuda_alloc_conf = cfg.get("pytorch_cuda_alloc_conf", None)
        if pytorch_cuda_alloc_conf:
            self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to: {pytorch_cuda_alloc_conf}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_cuda_alloc_conf

        # Set the seed for reproducibility
        self.seed = utils.set_seed(seed=cfg.seed)

        # Initialize the recipe state
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # Initialize various variables used in this recipe
        self._steps_per_epoch = None
        self._dataloader = None
        self._sampler = None
        self._loss_fn = None
        self._optimizer = None
        self._tokenizer = None
        self._model = None
        self._checkpointer = None
        self._lr_scheduler = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self):
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=self.cfg.checkpointer)
        self._model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
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
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=self.cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        utils.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )
        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
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
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        ds = self.dataset
        ds._tokenizer = self._tokenizer
        packed = cfg_dataset.get("packed", False)
        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ) if not packed else None,
        )
        return sampler, dataloader

    def save_checkpoint(self):
        ckpt_dict = {}
        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}
        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})
        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})
        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({utils.ADAPTER_CONFIG: adapter_config})
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=self.epochs_run,
        )

    def train(self):
        # Initialize tokens count and running loss (for grad accumulation)
        running_loss = 0
        num_tokens = 0
        t_step_start = time.perf_counter()

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            # Log per-epoch metrics and timestamps
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
                    self._lr_scheduler.step()
                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Log per-step metrics and timestamps
                    time_per_step = time.perf_counter() - t_step_start
                    mem_stats = utils.get_memory_stats(device=self._device)
                    self.scores_agent.log_step(
                        gpu_rank=0,  # Single device training
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
            self.scores_agent.log_epoch(gpu_rank=0, epoch_num=curr_epoch + 1, epoch_len=self.total_epochs,
                                        epoch_time_elapsed_s=time_per_epoch)
            self.epochs_run += 1


def recipe_main(cfg: DictConfig, dataset: Dataset, job_id: str, user_id: str):
    # Set up the main logger
    logger = setup_logger('torchtunewrapper_recipe', job_id, user_id)
    # A logger for logging scores; also propagates to main logger
    scores_logger = setup_logger('torchtunewrapper_recipe.metrics', job_id, user_id, add_stdout=False)
    # Setup logging and bigquery agent for scores
    scores_agent = TorchtunewrapperScoresAgent(job_id, scores_logger)
    # Initialize the recipe and start training
    recipe = LoRAFinetuneRecipeSingleDevice(cfg, logger, scores_agent, dataset)
    recipe.setup()
    recipe.train()
    recipe.save_checkpoint()
    recipe.cleanup()
