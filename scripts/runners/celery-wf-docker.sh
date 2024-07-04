#!/bin/bash

# Run a Celery workflow with the specified arguments using Docker
# Steps:
# 1. Log the start of the script
# 2. Read the version from the VERSION file
# 3. Define remote and local image names
# 4. Set the environment and image to use
# 5. Build or pull the Docker image based on the environment
# 6. Set GPU options based on OS type
# 7. Run the Docker container with appropriate volumes and environment variables
# 8. Log after the Docker container finishes

# Log start of the script
echo "Begin running the Celery workflow at $(date)"

# Import utility functions
source ./scripts/utils.sh

# Docker image
IMAGE_NAME="celery-workflow"
IMAGE_REMOTE_PREFIX="us-central1-docker.pkg.dev/neat-airport-407301/lum-docker-images/$IMAGE_NAME"
IMAGE_LOCAL="$IMAGE_NAME:$LOCAL_ENV"

# Read the version from the VERSION file
VERSION=$(cat VERSION)
echo "Read version $VERSION token: $PZ_HUGGINGFACE_TOKEN"

# Define remote and local image names
IMAGE_REMOTE="${IMAGE_REMOTE_PREFIX}:${VERSION}"

# Set the environment and image to use
image_use=$IMAGE_LOCAL
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  image_use=$IMAGE_REMOTE
fi

echo "Using image: $image_use"

# Build or pull the Docker image
if [[ "$image_use" == "$IMAGE_LOCAL" ]]; then
  echo "Building local Docker image"
  docker build -f celery.Dockerfile -t $image_use . > /dev/null 2>&1
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
docker run $gpus \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV=$PZ_ENV \
-e PZ_HUGGINGFACE_TOKEN=$PZ_HUGGINGFACE_TOKEN \
$image_use python pipeline/$1_wf.py "${@:2}"

# Log after Docker container finishes
echo "Celery workflow finished at $(date)"
