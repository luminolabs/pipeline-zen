#!/bin/bash

# Run a workflow with the specified arguments using Docker

set -e  # Exit immediately if a command fails

# Log start of the script
echo "Begin running the workflow at $(date)"

# Set the environment
source ./scripts/utils.sh 2>/dev/null || source /pipeline-zen-jobs/scripts/utils.sh 2>/dev/null

# Build the Docker image for the workflow
docker build -f workflows.Dockerfile --build-arg TARGET_WORKFLOW=$1 -t $1-workflow:$LOCAL_ENV .

# Set GPU options based on OS type
gpus="--gpus all"
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected macOS. Disabling GPU support for Docker"
  gpus=""
fi

# Log before running Docker container
echo "Running Docker container"

# Run the Docker container for the workflow
docker run $gpus \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV=$PZ_ENV \
-e PZ_DEVICE=$PZ_DEVICE \
-e PZ_RESULTS_BUCKET_SUFFIX=$PZ_RESULTS_BUCKET_SUFFIX \
-e PZ_HUGGINGFACE_TOKEN=$PZ_HUGGINGFACE_TOKEN \
-e PZ_CUSTOMER_API_KEY=$PZ_CUSTOMER_API_KEY \
-e HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1} \
$1-workflow:local python $1/cli.py "${@:2}"

# Log after Docker container finishes
echo "Workflow finished at $(date)"