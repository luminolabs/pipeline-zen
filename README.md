## Running with docker

### Running the train workflow

cd to this folder (repo root) and run:
```
docker build --build-arg TARGET_WORKFLOW=train -t train-workflow .
```
```
docker run --gpus all \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-v "$PWD/job_configs":/project/job_configs \
train-workflow imdb_sentiment
```
NOTE: `imdb_sentiment` above points to a file under `job_configs`

### Running the evaluate workflow

cd to this folder (repo root) and run:
```
docker build --build-arg TARGET_WORKFLOW=evaluate -t evaluate-workflow .
```
```
docker run --gpus all \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-v "$PWD/job_configs":/project/job_configs \
evaluate-workflow imdb_sentiment 2024-04-18-16-12-07.pt
```
NOTE: `imdb_sentiment` above points to a file under `job_configs` and
`2024-04-18-16-12-07.pt` points to the model weights file under `.results/model_weights/<job_id>`

IMPORTANT: Docker creates folders and files as root, so after running a workflow for the first time,
run the following command to change filesystem permissions back to your user:
```
sudo chown $(whoami) -R .results .cache .logs .secrets
```


## Running locally

Set PYTHONPATH to include `share-lib` and `job_configs`; cd to this folder (repo root) and run:
```
export PYTHONPATH=$(pwd)/shared-lib/src:$(pwd)
```

Download and copy the GCP service account credentials file to `.secrets` under the repo root. 
See this guide for instructions

For all workflows, ensure dependencies are installed; 
cd to the workflow's `src` (ex. `train/src`) folder and  run the following:
- `pip install -Ur ../requirements.txt`
- `pip install -Ur ../../shared-lib/requirements.txt`

### Running the train workflow

- cd to the `train/src` folder
- `python main.py mri_segmentation`

### Running the evaluate workflow

- cd to the `evaluate/src` folder
- `python main.py mri_segmentation 2024-04-18-16-57-23.pt`


## Outputs

Examples of workflow outputs

### Train Workflow

```
docker run --gpus all \                                           
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/job_configs":/project/job_configs \
-v $GOOGLE_APPLICATION_CREDENTIALS:/project/google_key.json:ro \
train-workflow mri_segmentation

train_workflow_metrics :: INFO :: System specs: {'gpu': [{'model': 'NVIDIA GeForce RTX 3080 Laptop GPU', 'memory': '16384 MiB', 'pwr_limit': '80.00 W'}], 'cpu': {'architecture': 'x86_64', 'cpus': '16', 'model_name': '11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz', 'threads_per_core': '2'}, 'mem': '31.21 GiB'}
train_workflow :: INFO :: Loading and configuring dataset!
train_workflow :: INFO :: Using `rainerberger/Mri_segmentation.train` from `huggingface`
Downloading readme: 100%|██████████| 522/522 [00:00<00:00, 2.38MB/s]
train_workflow :: INFO :: Dataset split has 400 records
train_workflow :: INFO :: Batch size is 8, number of batches is 50
train_workflow :: INFO :: Training on (cpu/cuda/mps?) device: cuda
train_workflow :: INFO :: Fetching the model
train_workflow :: INFO :: Using `single_label.unet` model
Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth" to /root/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth
100%|██████████| 74.4M/74.4M [00:02<00:00, 32.0MB/s]
train_workflow :: INFO :: Using `FocalLoss` loss function
train_workflow :: INFO :: Loss and Optimizer is set
train_workflow_metrics :: INFO :: Process started at: 2024-05-01 04:29:09
train_workflow_metrics :: INFO :: Batch #1/50, Loss: 0.3290
...
...
train_workflow_metrics :: INFO :: Batch #48/50, Loss: 0.1735
train_workflow_metrics :: INFO :: Batch #49/50, Loss: 0.1727
train_workflow_metrics :: INFO :: Batch #50/50, Loss: 0.1737
train_workflow_metrics :: INFO :: Epoch #3/3, Loss: 0.1737
train_workflow_metrics :: INFO :: Process ended at: 2024-05-01 04:30:56
train_workflow_metrics :: INFO :: Elapsed time: 1.77 minutes
train_workflow :: INFO :: Training loop complete, now saving the model
train_workflow :: INFO :: Trained model saved! at: ./.results/model_weights/mri-segmentation/2024-05-01-04-30-56.pt... use these arguments to evaluate your model: `mri_segmentation 2024-05-01-04-30-56.pt`
```

### Evaluate Workflow

```
docker run --gpus all \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/job_configs":/project/job_configs \
-v $GOOGLE_APPLICATION_CREDENTIALS:/project/google_key.json:ro \
evaluate-workflow mri_segmentation 2024-05-01-04-30-56.pt
evaluate_workflow_metrics :: INFO :: System specs: {'gpu': [{'model': 'NVIDIA GeForce RTX 3080 Laptop GPU', 'memory': '16384 MiB', 'pwr_limit': '80.00 W'}], 'cpu': {'architecture': 'x86_64', 'cpus': '16', 'model_name': '11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz', 'threads_per_core': '2'}, 'mem': '31.21 GiB'}
evaluate_workflow :: INFO :: Loading and configuring dataset!
evaluate_workflow :: INFO :: Using `rainerberger/Mri_segmentation.test` from `huggingface`
Downloading readme: 100%|██████████| 522/522 [00:00<00:00, 2.47MB/s]
evaluate_workflow :: INFO :: Dataset split has 100 records
evaluate_workflow :: INFO :: Batch size is 8, number of batches is 13
evaluate_workflow :: INFO :: Training on (cpu/cuda/mps?) device: cuda
evaluate_workflow :: INFO :: Fetching the model
evaluate_workflow :: INFO :: Using `single_label.unet` model
Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth" to /root/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth
100%|██████████| 74.4M/74.4M [00:02<00:00, 33.1MB/s]
evaluate_workflow :: INFO :: Using model weights path: ./.results/model_weights/mri-segmentation/2024-05-01-04-30-56.pt
evaluate_workflow_metrics :: INFO :: Process started at: 2024-05-01 04:34:57
evaluate_workflow :: INFO :: Batch 1/13
evaluate_workflow :: INFO :: Batch 2/13
...
...
evaluate_workflow :: INFO :: Batch 12/13
evaluate_workflow :: INFO :: Batch 13/13
evaluate_workflow_metrics :: INFO :: Stopped at batch: 14/13
Accuracy: 0.9927, 
Precision: 0.8080, 
Recall: 0.9958, 
F1: 0.7803
evaluate_workflow_metrics :: INFO :: Process ended at: 2024-05-01 04:35:01
evaluate_workflow_metrics :: INFO :: Elapsed time: 0.08 minutes
```
