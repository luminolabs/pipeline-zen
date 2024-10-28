#!/bin/bash

# Run a Celery workflow with the specified arguments

set -e  # Exit immediately if a command fails

# Use pipeline-zen root dir if provided
if [ -z "$PZ_ROOT_DIR" ]; then
  PZ_ROOT_DIR=$(pwd)
fi

# Set the paths to the Python modules
paths=$PZ_ROOT_DIR/lib-common/src
paths=$paths:$PZ_ROOT_DIR/lib-workflows/torchtunewrapper/src
paths=$paths:$PZ_ROOT_DIR/lib-celery/src

# Export .env environment variables
eval $(cat $PZ_ROOT_DIR/.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

# Export the variables so they're available to the python script
export PZ_ENV
export PZ_HUGGINGFACE_TOKEN
export PZ_CUSTOMER_API_KEY
export PZ_USE_MPS
export PZ_RESULTS_BUCKET_SUFFIX

# Run the Celery pipeline for the specified workflow
PYTHONPATH=$paths python $PZ_ROOT_DIR/lib-celery/src/pipeline/$1_wf.py "${@:2}"
