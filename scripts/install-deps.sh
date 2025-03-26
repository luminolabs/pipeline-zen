#!/bin/bash

# This script should be run locally only. It will install all dependencies required for the ML pipeline.
# Remote environments should use the Docker image to run the pipeline

# Upgrade pip
python -m pip install -U pip

# Install the required packages
python -m pip install -Ur lib-common/requirements.txt
python -m pip install -Ur lib-celery/requirements.txt
python -m pip install -Ur lib-workflows/torchtunewrapper/requirements.extra.txt
python -m pip install -Ur lib-workflows/torchtunewrapper/requirements.txt