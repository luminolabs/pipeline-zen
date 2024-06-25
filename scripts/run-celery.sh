#!/bin/bash

# Run a Celery workflow with the specified arguments

set -e  # Exit immediately if a command fails

# Set the paths to the Python modules
paths=$(pwd)/lib-common/src
paths=$paths:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src
paths=$paths:$(pwd)/lib-workflows/torchtunewrapper/src
paths=$paths:$(pwd)/lib-celery/src

# Run the Celery pipeline for the specified workflow
PYTHONPATH=$paths python lib-celery/src/pipeline/$1.py "${@:2}"
