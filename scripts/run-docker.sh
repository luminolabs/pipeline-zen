#!/bin/bash

# Run a workflow with the specified arguments using Docker

set -e  # Exit immediately if a command fails

# Build the Docker image for the workflow
docker build -f workflows.Dockerfile --build-arg TARGET_WORKFLOW=$1 -t $1-workflow:local .

# Set the environment name to use
env="local"
if [[ "$PZ_ENV" != "" && "$PZ_ENV" != "local" ]]; then
  env=$PZ_ENV
fi

# Set GPU options based on OS type
gpus="--gpus all"
if [[ "$OSTYPE" == "darwin"* ]]; then
  # There's no implementation on OSX to allow using the GPU with Docker;
  # this means that MPS will not be used
  # when running ML workflows on Docker under OSX (ie. the Mac GPU won't be used)
  gpus=""
fi

# Run the Docker container for the workflow
docker run $gpus \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV=$env \
$1-workflow:local python $1/cli.py "${@:2}"