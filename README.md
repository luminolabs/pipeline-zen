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
-v "$PWD/job_configs":/project/job_configs \
train-workflow alzheimermri_classification
```
NOTE: `alzheimermri_classification` above points to a file under `job_configs`

### Running the evaluate workflow

cd to this folder (repo root) and run:
```
docker build --build-arg TARGET_WORKFLOW=evaluate -t evaluate-workflow .
```
```
docker run --gpus all \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/job_configs":/project/job_configs \
evaluate-workflow alzheimermri_classification 2024-04-18-16-12-07.pt
```
NOTE: `alzheimermri_classification` above points to a file under `job_configs` and
`2024-04-18-16-12-07.pt` points to the model weights file under `.results/model_weights/<job_id>`

IMPORTANT: Docker creates folders and files as root, so after running a workflow for the first time,
run the following command to change filesystem permissions back to your user:
```
sudo chown $(whoami) -R .results .cache
```


## Running locally

Set PYTHONPATH to include `share-lib` and `job_configs`; cd to this folder (repo root) and run:
```
export PYTHONPATH=$(pwd)/shared-lib/src:$(pwd)
```

For all workflows, ensure dependencies are installed; cd to the workflow's `src` folder run the following:
- `pip install -Ur ../requirements.txt`
- `pip install -Ur ../../shared-lib/requirements.txt`

### Running the train workflow

- cd to the `train/src` folder
- `python main.py alzheimermri_classification`

### Running the evaluate workflow

- cd to the `evaluate/src` folder
- `python main.py alzheimermri_classification 2024-04-18-16-57-23.pt`


## Outputs

Examples of workflow outputs

### Train Workflow

```
$ docker run --gpus all -v "$PWD/.cache":/project/.cache -v "$PWD/.results":/project/.results train-workflow

Downloading readme:   0%|          | 0.00/2.13k [00:00<?, ?B/s]Loading and configuring dataset!
Downloading readme: 100%|██████████| 2.13k/2.13k [00:00<00:00, 14.0MB/s]
Downloading data: 100%|██████████| 22.6M/22.6M [00:00<00:00, 23.1MB/s]
Downloading data: 100%|██████████| 5.65M/5.65M [00:00<00:00, 22.2MB/s]
Generating train split: 100%|██████████| 5120/5120 [00:00<00:00, 120169.20 examples/s]
Generating test split: 100%|██████████| 1280/1280 [00:00<00:00, 193146.82 examples/s]
Training on (CPU/GPU?) device: cuda
Fetching the model
Loss and Optimizer is set
Training started at 2024-04-15 21:26:40
Epoch 1/10, Loss: 1.122030958160758
Epoch 2/10, Loss: 0.41792185306549073
Epoch 3/10, Loss: 0.17635376430116595
Epoch 4/10, Loss: 0.09475269303948153
Epoch 5/10, Loss: 0.0955704029853223
Epoch 6/10, Loss: 0.045235549423705376
Epoch 7/10, Loss: 0.050272475185556685
Epoch 8/10, Loss: 0.05423462625549291
Epoch 9/10, Loss: 0.03166594500580686
Epoch 10/10, Loss: 0.03592839671218826
Training ended at 2024-04-15 21:32:46
Total training time: 6.09 minutes
Training loop complete, now saving the model
Trained model saved!
```

### Evaluate Workflow

```
$ docker run -t --gpus all -v "$PWD/.cache":/project/.cache -v "$PWD/.results":/project/.results evaluate-workflow

Loading and configuring dataset!
Downloading readme: 100%|█████████████████████████████████████████| 2.13k/2.13k [00:00<00:00, 13.0MB/s]
Training on (CPU/GPU?) device: cuda
Fetching the model
config.json: 100%|████████████████████████████████████████████████| 69.6k/69.6k [00:00<00:00, 4.87MB/s]
model.safetensors: 100%|████████████████████████████████████████████| 102M/102M [00:02<00:00, 35.4MB/s]
Accuracy: 0.9673828125
Precision: 0.9375315617398208
Recall: 0.9708399900096027
F1 Score: 0.9513985583321037
```
