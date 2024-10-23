# Script to synchronize Torchtune recipe configs to Lumino

The script `sync_recipes.py` synchronizes (or rather, copies with modifications) torchtune recipe configs for various models into
their Lumino equivalent versions.

Look at the field `TORCHTUNE_RECIPE_MODELS` in the script to find
the list of whitelisted models which will be synchronized.

The script does not produce perfect output and a quick manual
verification is still needed.


## Script Help:

```
> python scripts/torchtune/sync_recipes.py --help
usage: sync_recipes.py [-h] -ttbp TORCHTUNE_BASE_PATH [-dest DESTINATION]

Sync torchtune recipe configs

options:
  -h, --help            show this help message and exit
  -ttbp TORCHTUNE_BASE_PATH, --torchtune_base_path TORCHTUNE_BASE_PATH
                        Base path of `torchtune` repo
  -dest DESTINATION, --destination DESTINATION
                        Base path of Lumino `torchtune` configs
```

## Example usage:

The following is an example usage 
```
> python scripts/torchtune/sync_recipes.py --torchtune_base_path ~/git/torchtune
src: /home/swami/git/torchtune/recipes/configs/mistral/7B_full.yaml, dest: job-configs/torchtune/mistral/7B_full.yml
src: /home/swami/git/torchtune/recipes/configs/mistral/7B_full_low_memory.yaml, dest: job-configs/torchtune/mistral/7B_full_low_memory.yml
src: /home/swami/git/torchtune/recipes/configs/mistral/7B_lora.yaml, dest: job-configs/torchtune/mistral/7B_lora.yml
src: /home/swami/git/torchtune/recipes/configs/mistral/7B_lora_single_device.yaml, dest: job-configs/torchtune/mistral/7B_lora_single_device.yml
src: /home/swami/git/torchtune/recipes/configs/mistral/7B_qlora_single_device.yaml, dest: job-configs/torchtune/mistral/7B_qlora_single_device.yml

```
