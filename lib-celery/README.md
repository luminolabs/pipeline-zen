# Celery workflows

### Example output of the `train_evaluate` workflow

```
./scripts/run-celery.sh \
  --job_config_name imdb_nlp_classification \
  --batch_size 8 \
  --num_epochs 2 \
  --num_batches 3
 
 
 -------------- celery@Vasiliss-MacBook-Pro.local v5.4.0 (opalescent)
--- ***** ----- 
-- ******* ---- macOS-14.4.1-arm64-arm-64bit 2024-05-05 17:45:55
- *** --- * --- 
- ** ---------- [config]
- ** ---------- .> app:         train_evaluate:0x13f08c210
- ** ---------- .> transport:   memory://localhost//
- ** ---------- .> results:     disabled://
- *** --- * --- .> concurrency: 12 (solo)
-- ******* ---- .> task events: OFF (enable -E to monitor tasks in this worker)
--- ***** ----- 
 -------------- [queues]
                .> celery           exchange=celery(direct) key=celery
                

[tasks]
  . train_evaluate.evaluate
  . train_evaluate.train

[2024-05-05 17:45:55,934: INFO/MainProcess] Connected to memory://localhost//
[2024-05-05 17:45:55,935: INFO/MainProcess] celery@Vasiliss-MacBook-Pro.local ready.


[2024-05-05 17:45:55,935: INFO/MainProcess] Task train_evaluate.train[3b70b1e3-b606-4cd0-bee5-081c985e4549] received
[2024-05-05 17:45:55,937: INFO/MainProcess] The job id is: imdb_nlp_classification-08814750-03bf-4101-93b7-57ead0339fb6
[2024-05-05 17:45:55,950: ERROR/MainProcess] `nvidia-smi` command not found
[2024-05-05 17:45:55,958: ERROR/MainProcess] `lscpu` command not found
[2024-05-05 17:45:55,967: ERROR/MainProcess] `/proc/meminfo` is not available in this system
[2024-05-05 17:45:55,967: INFO/MainProcess] System specs: {'gpu': None, 'cpu': None, 'mem': None}
[2024-05-05 17:45:56,442: INFO/MainProcess] Training job type: `JobCategory.NLP` - `JobType.CLASSIFICATION`
[2024-05-05 17:45:56,544: INFO/MainProcess] Loading and configuring dataset!
[2024-05-05 17:45:56,545: INFO/MainProcess] Using `stanfordnlp/imdb.train` from `huggingface`
[2024-05-05 17:46:00,300: INFO/MainProcess] Dataset split has 25000 records
[2024-05-05 17:46:00,300: INFO/MainProcess] Batch size is 8, number of batches is 3125
[2024-05-05 17:46:00,300: INFO/MainProcess] ...but only 3 batches are configured to run
[2024-05-05 17:46:00,301: INFO/MainProcess] Using `google-bert/bert-base-cased` tokenizer
[2024-05-05 17:46:00,425: INFO/MainProcess] Training on (cpu/cuda/mps?) device: mps
[2024-05-05 17:46:00,425: INFO/MainProcess] Fetching the model
[2024-05-05 17:46:01,062: INFO/MainProcess] Using `CrossEntropyLoss` loss function
[2024-05-05 17:46:01,063: INFO/MainProcess] Loss and Optimizer is set
[2024-05-05 17:46:01,063: INFO/MainProcess] Process started at: 2024-05-05-17-46-01
[2024-05-05 17:46:02,282: INFO/MainProcess] Batch #1/3125, Loss: 1.1100
[2024-05-05 17:46:03,380: INFO/MainProcess] Batch #2/3125, Loss: 2.4884
[2024-05-05 17:46:04,459: INFO/MainProcess] Batch #3/3125, Loss: 1.2218
[2024-05-05 17:46:04,536: INFO/MainProcess] Epoch #1/2, Loss: 0.0015
[2024-05-05 17:46:05,631: INFO/MainProcess] Batch #1/3125, Loss: 1.3205
[2024-05-05 17:46:06,747: INFO/MainProcess] Batch #2/3125, Loss: 1.4991
[2024-05-05 17:46:07,825: INFO/MainProcess] Batch #3/3125, Loss: 0.8641
[2024-05-05 17:46:07,924: INFO/MainProcess] Epoch #2/2, Loss: 0.0012
[2024-05-05 17:46:08,021: INFO/MainProcess] Process ended at: 2024-05-05-17-46-08
[2024-05-05 17:46:08,178: INFO/MainProcess] Elapsed time: 0.12 minutes
[2024-05-05 17:46:08,274: INFO/MainProcess] Training loop complete, now saving the model
[2024-05-05 17:46:09,075: INFO/MainProcess] Trained model saved! at: ./.results/model_weights/imdb_nlp_classification-08814750-03bf-4101-93b7-57ead0339fb6/2024-05-05-17-46-08.pt
[2024-05-05 17:46:09,077: INFO/MainProcess] Task train_evaluate.train[3b70b1e3-b606-4cd0-bee5-081c985e4549] succeeded in 13.14148199999181s: 'imdb_nlp_classification-08814750-03bf-4101-93b7-57ead0339fb6/2024-05-05-17-46-08.pt'


[2024-05-05 17:46:09,077: INFO/MainProcess] Task train_evaluate.evaluate[a271836d-5cc9-44b5-9f6c-80f64936be54] received
[2024-05-05 17:46:09,078: INFO/MainProcess] The job id is: imdb_nlp_classification-08814750-03bf-4101-93b7-57ead0339fb6-2024-05-05-17-46-09
[2024-05-05 17:46:09,107: ERROR/MainProcess] `nvidia-smi` command not found
[2024-05-05 17:46:09,132: ERROR/MainProcess] `lscpu` command not found
[2024-05-05 17:46:09,158: ERROR/MainProcess] `/proc/meminfo` is not available in this system
[2024-05-05 17:46:09,158: INFO/MainProcess] System specs: {'gpu': None, 'cpu': None, 'mem': None}
[2024-05-05 17:46:09,562: INFO/MainProcess] Training job type: `JobCategory.NLP` - `JobType.CLASSIFICATION`
[2024-05-05 17:46:09,661: INFO/MainProcess] Loading and configuring dataset!
[2024-05-05 17:46:09,661: INFO/MainProcess] Using `stanfordnlp/imdb.test` from `huggingface`
[2024-05-05 17:46:12,848: INFO/MainProcess] Dataset split has 25000 records
[2024-05-05 17:46:12,848: INFO/MainProcess] Batch size is 8, number of batches is 3125
[2024-05-05 17:46:12,848: INFO/MainProcess] ...but only 3 batches are configured to run
[2024-05-05 17:46:12,848: INFO/MainProcess] Using `google-bert/bert-base-cased` tokenizer
[2024-05-05 17:46:12,960: INFO/MainProcess] Training on (cpu/cuda/mps?) device: mps
[2024-05-05 17:46:12,960: INFO/MainProcess] Fetching the model
[2024-05-05 17:46:12,960: INFO/MainProcess] Using `single_label.cardiffnlp/twitter-roberta-base-sentiment-latest` model
[2024-05-05 17:46:13,841: INFO/MainProcess] Using model weights path: ./.results/model_weights/imdb_nlp_classification-08814750-03bf-4101-93b7-57ead0339fb6/2024-05-05-17-46-08.pt
[2024-05-05 17:46:14,026: INFO/MainProcess] Process started at: 2024-05-05-17-46-14
[2024-05-05 17:46:14,427: INFO/MainProcess] Batch 1/3125
[2024-05-05 17:46:14,721: INFO/MainProcess] Batch 2/3125
[2024-05-05 17:46:15,015: INFO/MainProcess] Batch 3/3125
[2024-05-05 17:46:15,015: INFO/MainProcess] Reached `num_batches` limit: 3
[2024-05-05 17:46:15,019: INFO/MainProcess] Stopped at batch: 3/3125
Accuracy: 0.4167, 
Precision: 0.5000, 
Recall: 0.2083, 
F1: 0.2941
[2024-05-05 17:46:15,120: INFO/MainProcess] Process ended at: 2024-05-05-17-46-15
[2024-05-05 17:46:15,219: INFO/MainProcess] Elapsed time: 0.02 minutes
[2024-05-05 17:46:15,317: INFO/MainProcess] Task train_evaluate.evaluate[a271836d-5cc9-44b5-9f6c-80f64936be54] succeeded in 6.23928775000968s: (0.4166666666666667, 0.5, 0.20833333333333334, 0.29411764705882354)
```