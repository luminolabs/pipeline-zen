# Fine-tuning LLMs with Torchtune

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

### Run the torchtunewrapper workflow

Note: Unless you have a beefy machine, you probably want to run this workflow remotely, 
not locally; see the next section for instructions.

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