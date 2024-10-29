#!/bin/bash

# Run a Celery workflow with the specified arguments

set -e  # Exit immediately if a command fails

# If the pipeline-zen root dir is provided, change directory to it
if [ -n "$PZ_ROOT_DIR" ]; then
  cd $PZ_ROOT_DIR
fi

# Set this to current directory if not set
if [ -z "$PZ_ENV_DIR" ]; then
  PZ_ENV_DIR=$(pwd)
fi

# Set the paths to the Python modules
paths=./lib-common/src
paths=$paths:./lib-workflows/torchtunewrapper/src
paths=$paths:./lib-celery/src

# Export .env environment variables
set -o allexport
eval $(cat $PZ_ENV_DIR/.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

# Run the Celery pipeline for the specified workflow
PYTHONPATH=$paths python ./lib-celery/src/pipeline/$1_wf.py "${@:2}"
