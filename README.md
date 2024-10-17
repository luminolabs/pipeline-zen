# Fine-tuning LLMs with Torchtune

## Development
Notes:
- There is support for single-node and multi-gpu training but not multi-node multi-gpu training yet.
- Training recipes are under `lib-workflows/torchtunewrapper/recipes/` and are copied, modified, and simplified from the official `torchtune` repo.
- Check out [DEPLOY.md](DEPLOY.md) for instructions on how to deploy the pipeline.

## Running remotely on a VM

### Postman

Look in Postman for the [Create Job](https://lumino-labs.postman.co/workspace/Scheduler-API~d706ab0f-5da2-4197-89f0-ebaf9c8d4d53/request/37668647-ca3ae092-fd3c-406d-97a3-3de39ffb4af1?action=share&source=copy-link&creator=37668647&active-environment=708ca9aa-c49f-47f0-ad4a-bd37195418cc) template.

## Running locally

Depending on your machine type and specs, you will probably want to run this workflow remotely, not locally.

### Setup

Download and copy the GCP service account credentials file to `.secrets` under the repo root.
[Follow this guide for instructions.](https://www.notion.so/luminoai/Create-a-GCP-credentials-file-for-pipeline-zen-d2a007730f204ae797db8c0174224ddc)

...or use the following command to authenticate with GCP:

```bash
gcloud auth application-default login
```

Install python dependencies:

```bash
./scripts/install-deps.sh
```

### Run the dummy torchtunewrapper workflow

This will run the pipeline with a dummy dataset and dummy model, but it won't actually train anything. 
It's useful for testing the pipeline and making sure everything is set up correctly, or when you want to test
integration with the scheduler or the protocol.

All you have to do is set the `--job_config_name` to `llm_dummy`.

```bash
./scripts/runners/celery-wf.sh torchtunewrapper --job_config_name llm_dummy --job_id -1 \
  --dataset_id gs://lum-pipeline-zen-jobs-us/datasets/protoml/text2sql.jsonl \
  --batch_size 2 --shuffle true --num_epochs 1 --use_lora false \
  --num_gpus 1 --user_id -1 --lr 1e-2 --seed 1234 
````

### Run the torchtunewrapper workflow

Note: Unless you have a beefy machine, you probably want to run this workflow remotely, 
not locally; see top of this doc for instructions.

```bash
./scripts/runners/celery-wf.sh torchtunewrapper \
  --job_config_name llm_llama3_1_8b \
  --job_id llm_llama3_1_8b-experiment1 \
  --dataset_id gs://lum-pipeline-zen-jobs-us/datasets/protoml/text2sql.jsonl \
  --batch_size 2 --shuffle true --num_epochs 1 \
  --use_lora true --use_qlora false \
  --lr 3e-4 --seed 42 \
  --num_gpus 1
```
