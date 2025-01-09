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