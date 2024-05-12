#!/bin/bash

pip install -Ur lib-common/requirements.txt
pip install -Ur lib-common/requirements-test.txt
pip install -Ur lib-celery/requirements.txt
pip install -Ur lib-workflows/train/requirements.txt
pip install -Ur lib-workflows/evaluate/requirements.txt
pip install -Ur scripts/requirements.txt
