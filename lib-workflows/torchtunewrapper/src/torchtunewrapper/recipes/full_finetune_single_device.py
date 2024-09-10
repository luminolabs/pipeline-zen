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
from torchtune.recipe_interfaces import FTRecipeInterface

from common.agents.model_scores import TorchtunewrapperScoresAgent
from common.utils import setup_logger


# noinspection PyProtocol
class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full fine-tuning recipe for single device training.
    """
    def __init__(self, cfg: DictConfig,
                 logger: Logger, scores_agent: TorchtunewrapperScoresAgent, dataset: Dataset):
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

        # Training configuration
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.optimizer_in_bwd

        # Set the PyTorch CUDA allocation configuration
        # This is useful for memory management on GPUs and can be used to prevent OOM errors
        pytorch_cuda_alloc_conf = cfg.get("pytorch_cuda_alloc_conf", None)
        if pytorch_cuda_alloc_conf:
            self.logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF to: {pytorch_cuda_alloc_conf}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_cuda_alloc_conf

        if self._gradient_accumulation_steps > 1 and self._optimizer_in_bwd:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

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
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self):
        ckpt_dict = self.load_checkpoint(self.cfg.checkpointer)
        self._model = self._setup_model(
            cfg_model=self.cfg.model,
            enable_activation_checkpointing=self.cfg.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(self.cfg.tokenizer)
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=self.cfg.optimizer,
            optimizer_in_bwd=self.cfg.optimizer_in_bwd,
            opt_state_dict=(
                ckpt_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None
            ),
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
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        model.load_state_dict(model_state_dict)
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
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
                for p in self._model.parameters()
            }
            # Register optimizer step hooks on the model to run optimizer in backward.
            utils.register_optim_in_bwd_hooks(model=self._model, optim_dict=optim_dict)
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = utils.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # use optimizer in backward.
            if opt_state_dict is not None:
                try:
                    self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)
            return optimizer

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
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ) if not packed else None,
        )
        return sampler, dataloader

    def save_checkpoint(self):
        ckpt_dict = {utils.MODEL_KEY: self._model.state_dict()}
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=self.epochs_run,
            intermediate_checkpoint=(self.epochs_run < self.total_epochs),
        )

    def train(self):
        if not self._optimizer_in_bwd:
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
                    if not self._optimizer_in_bwd:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    # Log per-step metrics and timestamps
                    time_per_step = time.perf_counter() - t_step_start
                    mem_stats = utils.get_memory_stats(device=self._device)
                    self.scores_agent.log_step(
                        gpu_rank=0,  # single device training
                        step_num=self.global_step,
                        step_len=self._steps_per_epoch,
                        step_loss=running_loss.item(),
                        step_lr=(
                            self._optim_ckpt_wrapper.get_optim_key("lr")
                            if self._optimizer_in_bwd
                            else self._optimizer.param_groups[0]["lr"]
                        ),
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
            self.scores_agent.log_epoch(
                gpu_rank=0,
                epoch_num=curr_epoch + 1,
                epoch_len=self.total_epochs,
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
    recipe = FullFinetuneRecipeSingleDevice(cfg, logger, scores_agent, dataset)
    recipe.setup()
    recipe.train()
    recipe.save_checkpoint()
    recipe.cleanup()
