### How to run a workflow

- cd to this folder
- `docker build --build-arg TARGET_WORKFLOW=train -t train-workflow .`
- `docker run --gpus all -v "$PWD/.cache":/project/.cache -v "$PWD/.results":/project/.results train-workflow`

replace `train` above with either `train` or `evaluate` to specify the workflow

Note: `print` statements are delayed and flushed later when run in docker