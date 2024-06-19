#!/bin/bash

# Log start of the script
echo "Script started at $(date)"

# Read the version from the VERSION file
VERSION=$(cat VERSION)
echo "Read version: $VERSION"

# Define remote and local image names
image_remote=us-central1-docker.pkg.dev/neat-airport-407301/lum-docker-images/celery-workflow:$VERSION
image_local=celery-workflow:local

# Set the environment and image to use
env="local"
image_use=$image_local
if [[ "$PZ_ENV" != "local" && "$PZ_ENV" != "" ]]; then
  env=$PZ_ENV
  image_use=$image_remote
  echo "Environment is not local. Changing directory to /pipeline-zen-jobs"
  cd /pipeline-zen-jobs || { echo "Failed to change directory to /pipeline-zen-jobs"; exit 1; }
fi

echo "Using image: $image_use"

# Build or pull the Docker image
if [[ "$image_use" == "$image_local" ]]; then
  echo "Building local Docker image"
  docker build -f celery.Dockerfile -t $image_use .
else
  echo "Pulling remote Docker image"
  docker pull $image_use
fi

# Set GPU options based on OS type
gpus="--gpus all"
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected macOS. Disabling GPU support for Docker"
  gpus=""
fi

# Log before running Docker container
echo "Running Docker container with image: $image_use"

# Run the Docker container
#docker run $gpus \
#-v "$PWD/.cache":/project/.cache \
#-v "$PWD/.results":/project/.results \
#-v "$PWD/.logs":/project/.logs \
#-v "$PWD/.secrets":/project/.secrets \
#-e PZ_ENV=$env \
#-e PZ_HUGGINGFACE_TOKEN=$PZ_HUGGINGFACE_TOKEN \
#$image_use python pipeline/$1_wf.py "${@:2}"
sleep 60

# Log after Docker container finishes
echo "Celery workflow finished at $(date)"

# Log end of the script
echo "Script ended at $(date)"
