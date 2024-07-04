#!/bin/bash

# Run a workflow with the specified arguments

set -e  # Exit immediately if a command fails

# Set the paths to the Python modules
paths=$(pwd)/lib-common/src:$(pwd)/lib-workflows/$1/src

# Run the script for the specified workflow
PYTHONPATH=$paths python lib-workflows/$1/src/$1/cli.py "${@:2}"
