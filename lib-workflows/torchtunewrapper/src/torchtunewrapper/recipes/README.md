# Torchtunewrapper Recipes

This directory contains the fine-tuning recipes used by the Pipeline Zen ML training pipeline. 

The recipes are adapted from the official [Torchtune](https://github.com/pytorch/torchtune) repository and modified for use with our pipeline architecture.

## Recipe Types

The following fine-tuning recipes are available:

### Core Training Recipes
- `full_finetune_distributed.py` - Full model fine-tuning using distributed training (multi-GPU on single node)
- `full_finetune_single_device.py` - Full model fine-tuning on a single GPU
- `lora_finetune_distributed.py` - LoRA fine-tuning using distributed training (multi-GPU on single node)
- `lora_finetune_single_device.py` - LoRA fine-tuning on a single GPU

### Testing Recipe
- `dummy.py` - A test recipe that simulates the training process without actually training anything. Useful for testing the pipeline infrastructure and integration points.

## Recipe Base Class

All recipes inherit from `recipe_base.py` which provides:

- Common initialization logic
- Job logging and monitoring capabilities
- Heartbeat functionality
- Standard methods that must be implemented by each recipe:
    - `_init()` - Recipe initialization
    - `_load_checkpoint()` - Load model checkpoints
    - `_setup()` - Setup training components
    - `_train()` - Core training loop
    - `_save_checkpoint()` - Save model checkpoints
    - `_cleanup()` - Cleanup after training

## Key Features

The recipes support:

- Multiple precision types (fp32, bf16)
- Activation checkpointing for memory efficiency
- Activation offloading to CPU
- Different optimizer configurations
- Gradient accumulation
- Gradient clipping
- Custom learning rate scheduling
- Performance profiling
- Metric logging
- Model checkpointing

## Architecture Notes

1. All recipes integrate with the Pipeline Zen logging and monitoring system via the `TorchtunewrapperLoggerAgent`
2. Training configurations are loaded from yaml files in the `job-configs/torchtune/` directory
3. Checkpoints and artifacts are automatically uploaded to cloud storage when running in production

## Usage

The recipes are not meant to be used directly but rather through the Pipeline Zen workflow system. See the main README.md for details on running training jobs.

Example workflow command:
```bash
./scripts/runners/celery-wf.sh torchtunewrapper \
  --job_config_name llm_llama3_1_8b \
  --dataset_id <dataset_path> \
  --batch_size 2 \
  --num_epochs 1 \
  --use_lora true \
  --num_gpus 1
```

## Porting Torchtune Recipes

To port a recipe from Torchtune to the Pipeline Zen system:

1. Required Imports:
```python
from torch.utils.data import Dataset
from torchtunewrapper.recipes.recipe_base import RecipeBase
from torchtunewrapper.utils import run_recipe
```

2. Recipe Class Changes:
- Inherit from `RecipeBase` instead of `FTRecipeInterface`
- Rename `__init__` method to `_init`
- Add underscore prefix to all abstract method names defined in `RecipeBase`
- Remove the `if __name__ == "__main__":` block

3. Modify `recipe_main`:
- Remove `@config.parse` decorator
- Update signature:
```python
def recipe_main(job_id: str, user_id: str, cfg: DictConfig, dataset: Dataset) -> None:
```
- Replace recipe instantiation with:
```python
run_recipe(RecipeClassName, job_id, user_id, cfg, dataset)
```

4. Update `_train` method:
- Remove checkpoint saving code
- Add epoch timing:
```python
t_epoch_start = time.perf_counter()  # Add above t0 declaration
```
- Add Pipeline Zen logging after metrics logging but before t0 reset:
```python
# Lumino specific logging / per step
self.job_logger.log_step(
    gpu_rank=self.rank,
    step_num=self.global_step,
    step_len=self._steps_per_epoch * self.total_epochs,
    step_loss=log_dict["loss"],
    step_lr=log_dict["lr"],
    step_peak_memory_active=log_dict.get("peak_memory_active"),
    step_peak_memory_alloc=log_dict.get("peak_memory_alloc"),
    step_peak_memory_reserved=log_dict.get("peak_memory_reserved"),
    step_time_elapsed_s=time_per_step,
    epoch_num=curr_epoch + 1,
    epoch_len=self.total_epochs,
)
```
- Add epoch logging after `self.epochs_run += 1`:
```python
# Lumino specific logging / per epoch
time_per_epoch = time.perf_counter() - t_epoch_start
self.job_logger.log_epoch(gpu_rank=self.rank, epoch_num=self.epochs_run,
                          epoch_len=self.total_epochs,
                          epoch_time_elapsed_s=time_per_epoch)
t_epoch_start = time.perf_counter()
```

5. Update `_setup_data`:
   Add dataset handling at start of method:
```python
if self.dataset:
    ds = self.dataset
    packed = cfg_dataset.get("packed", False)
elif ... # rest of existing if block
```

6. Code Cleanup:
- Format code (IntelliJ: Cmd+Ctrl+L)
- Optimize imports (IntelliJ: Alt+Ctrl+O)

## Recipe Development

When developing new recipes or modifying existing ones:

1. Inherit from `RecipeBase` class
2. Implement all required abstract methods
3. Use the provided logging and monitoring capabilities
4. Ensure compatibility with both single-device and distributed training if needed
5. Add appropriate configuration files in `job-configs/torchtune/`
6. Test with both the Celery workflow runner and direct execution

## Limitations

Current limitations of the recipe system:

- No multi-node training support yet
- Limited to PyTorch models