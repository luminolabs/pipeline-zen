#!/bin/bash

# This script should be run locally only. It will install all dependencies required for the ML pipeline.
# Remote environments should use the Docker image to run the pipeline

pip install -Ur lib-common/requirements.txt
pip install -Ur lib-common/requirements-test.txt
pip install -Ur lib-workflows/train/requirements.txt
pip install -Ur lib-workflows/evaluate/requirements.txt
pip install -Ur lib-workflows/torchtunewrapper/requirements.txt
pip install -Ur lib-celery/requirements.txt
pip install -Ur scripts/requirements.txt
