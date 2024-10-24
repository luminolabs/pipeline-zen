#!/bin/bash

# Run a workflow with the specified arguments

set -e  # Exit immediately if a command fails

# Set the paths to the Python modules
paths=$(pwd)/lib-common/src:$(pwd)/lib-workflows/$1/src

# Export .env environment variables
eval $(cat ./.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

# Export the variables so they're available to the python script
export PZ_ENV
export PZ_HUGGINGFACE_TOKEN
export PZ_CUSTOMER_API_KEY
export PZ_USE_MPS
export PZ_RESULTS_BUCKET_SUFFIX

# Run the script for the specified workflow
PYTHONPATH=$paths python lib-workflows/$1/src/$1/cli.py "${@:2}"
