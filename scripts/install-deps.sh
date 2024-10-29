#!/bin/bash

# This script should be run locally only. It will install all dependencies required for the ML pipeline.
# Remote environments should use the Docker image to run the pipeline

# Upgrade pip
pip install -U pip

# Install the required packages
pip install -Ur lib-common/requirements.txt
pip install -Ur lib-celery/requirements.txt
pip install -Ur lib-workflows/torchtunewrapper/requirements.extra.txt
pip install -Ur lib-workflows/torchtunewrapper/requirements.txt