#!/bin/bash

# Run a Celery workflow with the specified arguments using Docker

set -e  # Exit immediately if a command fails

# Log start of the script
echo "Begin running the Celery workflow at $(date)"

# Set the environment
source ./scripts/utils.sh 2>/dev/null || source /pipeline-zen-jobs/scripts/utils.sh 2>/dev/null

# Docker image
IMAGE_NAME="celery-workflow"
IMAGE_REMOTE_PREFIX="us-central1-docker.pkg.dev/neat-airport-407301/lum-docker-images/$IMAGE_NAME"
IMAGE_LOCAL="$IMAGE_NAME:$LOCAL_ENV"

# Read the version from the VERSION file
VERSION=$(cat VERSION)
echo "Read version $VERSION"

# Define remote and local image names
IMAGE_REMOTE="${IMAGE_REMOTE_PREFIX}:${VERSION}"

# Set the environment and image to use
image_use=$IMAGE_LOCAL
if [[ "$PZ_ENV" != "$LOCAL_ENV" && "$PZ_ENV" != "cpnode" ]]; then
  image_use=$IMAGE_REMOTE
fi

echo "Using image: $image_use"

# Force linux/amd64 platform for Docker for local runs,
# as there are issues with torchao when running natively on apple silicon
docker_arch=""
# Build the Docker image if running locally
if [[ "$image_use" == "$IMAGE_LOCAL" ]]; then
  docker_arch="--platform linux/amd64"
  echo "Building local Docker image"
  docker build -f celery.Dockerfile -t $image_use $docker_arch . > /dev/null 2>&1
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
docker run $docker_arch $gpus \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV=$PZ_ENV \
-e PZ_DEVICE=$PZ_DEVICE \
-e PZ_RESULTS_BUCKET_SUFFIX=$PZ_RESULTS_BUCKET_SUFFIX \
-e PZ_HUGGINGFACE_TOKEN=$PZ_HUGGINGFACE_TOKEN \
-e PZ_CUSTOMER_API_KEY=$PZ_CUSTOMER_API_KEY \
-e HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1} \
$image_use python pipeline/$1_wf.py "${@:2}"

# Log after Docker container finishes
echo "Celery workflow finished at $(date)"
