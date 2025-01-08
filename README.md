# Fine-tuning LLMs with Torchtune

## Development
Notes:
- There is support for single-node and multi-gpu training but not multi-node multi-gpu training yet.
- Training recipes are under `lib-workflows/torchtunewrapper/src/torchtunewrapper/recipes/` and are copied, modified, and simplified from the official `torchtune` repo.
- Check out [DEPLOY.md](DEPLOY.md) for instructions on how to deploy the pipeline.

## Running remotely on a VM

### Postman

Look in Postman for the `Create Job (GCP)` request under `Scheduler API` to see how to create a job.

## Running locally

Depending on your machine type and specs, you will most likely want to run this workflow remotely, not locally.

### Setup

Create .env file in the root directory. Add below keys to .env

```bash
PZ_ENV=local
PZ_DEVICE=cpu
PZ_HUGGINGFACE_TOKEN=<HF-token>
PZ_CUSTOMER_API_KEY=<api-key>
```

Place a Google application default credentials file under `secrets/gcp_key.json`.

Download the key from the GCP console, look for the `pipeline-zen-jobs-dev` service account, and create a key for it.

Install python dependencies:

```bash
./scripts/install-deps.sh
```

Note: This step might not be needed anymore. Proceed without it and see if it works.
On a Mac, you to install torchao in a specific way, clone it outside the `pipeline-zen` repo:

```bash
git clone https://github.com/pytorch/ao
cd ao
git checkout v0.3.1-rc1
python setup.py install
TORCHAO_NIGHTLY=1 python setup.py install
```

Note: Virtualenv should be active before running the following commands in the ao directory

```bash
pip install "torchtune==0.2.1"
pip install --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Run the dummy torchtunewrapper workflow

This will run the pipeline with a dummy dataset and dummy model, but it won't actually train anything. 
It's useful for testing the pipeline and making sure everything is set up correctly, or when you want to test
integration with the scheduler or the protocol.

All you have to do is set the `--job_config_name` to `llm_dummy`.

```bash
./scripts/runners/celery-wf.sh torchtunewrapper \
  --job_config_name llm_dummy \
  --job_id llm_dummy-experiment1 \
  --dataset_id gs://lum-dev-pipeline-zen-datasets/0ca98b07-9366-4a31-8c83-569961c90294/2024-12-17_21-57-21_text2sql.jsonl \
  --batch_size 2 --shuffle true --num_epochs 1 \
  --use_lora true --use_qlora false \
  --lr 3e-4 --seed 42 \
  --num_gpus 1
````

### Run the actual torchtunewrapper workflow

Note: Don't run this unless you have a machine with NVIDIA GPUs and the necessary drivers and CUDA installed.

```bash
./scripts/runners/celery-wf.sh torchtunewrapper \
  --job_config_name llm_llama3_1_8b \
  --job_id llm_llama3_1_8b-experiment1 \
  --dataset_id gs://lum-dev-pipeline-zen-datasets/0ca98b07-9366-4a31-8c83-569961c90294/2024-12-17_21-57-21_text2sql.jsonl \
  --batch_size 2 --shuffle true --num_epochs 1 \
  --use_lora true --use_qlora false \
  --lr 3e-4 --seed 42 \
  --num_gpus 1
```
