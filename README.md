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
train-workflow --job_config_name imdb_nlp_classification
```
NOTE: `imdb_nlp_classification` above points to a file under `job_configs`

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
evaluate-workflow \
    --job_config_name imdb_nlp_classification \
    --model_weights imdb_nlp_classification/2024-05-02-15-09-31.pt
```
NOTE: `imdb_nlp_classification` above points to a file under `job_configs` and
`imdb_nlp_classification-experiment1/2024-05-02-15-09-31.pt` points to the 
model weights file under `.results/model_weights/`

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
[Follow this guide for instructions.](https://www.notion.so/luminoai/Create-a-GCP-credentials-file-for-pipeline-zen-d2a007730f204ae797db8c0174224ddc)

For all workflows, ensure dependencies are installed; 
cd to the workflow's `src` (ex. `train/src`) folder and  run the following:
- `pip install -Ur ../requirements.txt`
- `pip install -Ur ../../shared-lib/requirements.txt`

### Running the train workflow

- cd to the `train/src` folder
- `python main.py --job_config_name imdb_nlp_classification`

### Running the evaluate workflow

- cd to the `evaluate/src` folder
```
python main.py \
    --job_config_name imdb_nlp_classification \
    --model_weights imdb_nlp_classification/2024-05-02-15-09-31.pt
```


## Outputs

Examples of workflow outputs

### Train Workflow

```
$ python main.py --job_config_name imdb_nlp_classification --batch_size 8 --num_epochs 2 --num_batches 3

train_workflow :: INFO :: The job id is: imdb_nlp_classification-2024-05-02-15-31-14
train_workflow_metrics :: ERROR :: `nvidia-smi` command not found
train_workflow_metrics :: ERROR :: `lscpu` command not found
train_workflow_metrics :: ERROR :: `/proc/meminfo` is not available in this system
train_workflow_metrics :: INFO :: System specs: {'gpu': None, 'cpu': None, 'mem': None}
train_workflow_metrics :: INFO :: Training job type: `JobCategory.NLP` - `JobType.CLASSIFICATION`
train_workflow :: INFO :: Loading and configuring dataset!
train_workflow :: INFO :: Using `stanfordnlp/imdb.train` from `huggingface`
train_workflow :: INFO :: Dataset split has 25000 records
train_workflow :: INFO :: Batch size is 8, number of batches is 3125
train_workflow :: INFO :: ...but only 3 batches are configured to run
train_workflow :: INFO :: Using `google-bert/bert-base-cased` tokenizer
train_workflow :: INFO :: Training on (cpu/cuda/mps?) device: mps
train_workflow :: INFO :: Fetching the model
train_workflow :: INFO :: Using `single_label.cardiffnlp/twitter-roberta-base-sentiment-latest` model
Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
train_workflow :: INFO :: Using `CrossEntropyLoss` loss function
train_workflow :: INFO :: Loss and Optimizer is set
train_workflow_metrics :: INFO :: Process started at: 2024-05-02-15-31-19
train_workflow_metrics :: INFO :: Batch #1/3125, Loss: 1.2770
train_workflow_metrics :: INFO :: Batch #2/3125, Loss: 1.5317
train_workflow_metrics :: INFO :: Batch #3/3125, Loss: 1.0876
train_workflow_metrics :: INFO :: Epoch #1/2, Loss: 0.0012
train_workflow_metrics :: INFO :: Batch #1/3125, Loss: 1.4425
train_workflow_metrics :: INFO :: Batch #2/3125, Loss: 1.0437
train_workflow_metrics :: INFO :: Batch #3/3125, Loss: 0.9804
train_workflow_metrics :: INFO :: Epoch #2/2, Loss: 0.0011
train_workflow_metrics :: INFO :: Process ended at: 2024-05-02-15-31-26
train_workflow_metrics :: INFO :: Elapsed time: 0.12 minutes
train_workflow :: INFO :: Training loop complete, now saving the model
train_workflow :: INFO :: Trained model saved! at: ../../.results/model_weights/imdb_nlp_classification-2024-05-02-15-31-14/2024-05-02-15-31-26.pt
```

### Evaluate Workflow

```
$ python main.py --job_config_name imdb_nlp_classification --model_weights imdb_nlp_classification-2024-05-02-15-31-14/2024-05-02-15-31-26.pt --batch_size 8 --num_batches 3

evaluate_workflow :: INFO :: The job id is: imdb_nlp_classification-2024-05-02-15-32-38
evaluate_workflow_metrics :: ERROR :: `nvidia-smi` command not found
evaluate_workflow_metrics :: ERROR :: `lscpu` command not found
evaluate_workflow_metrics :: ERROR :: `/proc/meminfo` is not available in this system
evaluate_workflow_metrics :: INFO :: System specs: {'gpu': None, 'cpu': None, 'mem': None}
evaluate_workflow_metrics :: INFO :: Training job type: `JobCategory.NLP` - `JobType.CLASSIFICATION`
evaluate_workflow :: INFO :: Loading and configuring dataset!
evaluate_workflow :: INFO :: Using `stanfordnlp/imdb.test` from `huggingface`
evaluate_workflow :: INFO :: Dataset split has 25000 records
evaluate_workflow :: INFO :: Batch size is 8, number of batches is 3125
evaluate_workflow :: INFO :: ...but only 3 batches are configured to run
evaluate_workflow :: INFO :: Using `google-bert/bert-base-cased` tokenizer
evaluate_workflow :: INFO :: Training on (cpu/cuda/mps?) device: mps
evaluate_workflow :: INFO :: Fetching the model
evaluate_workflow :: INFO :: Using `single_label.cardiffnlp/twitter-roberta-base-sentiment-latest` model
Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
evaluate_workflow :: INFO :: Using model weights path: ../../.results/model_weights/imdb_nlp_classification-2024-05-02-15-31-14/2024-05-02-15-31-26.pt
evaluate_workflow_metrics :: INFO :: Process started at: 2024-05-02-15-32-43
evaluate_workflow :: INFO :: Batch 1/3125
evaluate_workflow :: INFO :: Batch 2/3125
evaluate_workflow :: INFO :: Batch 3/3125
evaluate_workflow :: INFO :: Reached `num_batches` limit: 3
/Users/vasilis/venv/luminoai/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1509: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
evaluate_workflow_metrics :: INFO :: Stopped at batch: 3/3125
Accuracy: 0.5833, 
Precision: 0.5000, 
Recall: 0.2917, 
F1: 0.3684
evaluate_workflow_metrics :: INFO :: Process ended at: 2024-05-02-15-32-44
evaluate_workflow_metrics :: INFO :: Elapsed time: 0.02 minutes
```
