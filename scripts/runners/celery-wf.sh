#!/bin/bash

# Run a Celery workflow with the specified arguments

set -e  # Exit immediately if a command fails

# Set the paths to the Python modules
paths=$(pwd)/lib-common/src
paths=$paths:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src
paths=$paths:$(pwd)/lib-workflows/torchtunewrapper/src
paths=$paths:$(pwd)/lib-celery/src

# Export .env environment variables
eval $(cat ./.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

# Export the variables so they're available to the python script
export PZ_ENV
export PZ_HUGGINGFACE_TOKEN
export PZ_CUSTOMER_API_KEY

# Run the Celery pipeline for the specified workflow
PYTHONPATH=$paths python lib-celery/src/pipeline/$1_wf.py "${@:2}"
