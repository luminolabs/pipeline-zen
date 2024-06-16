#!/bin/bash

paths=$(pwd)/lib-common/src
paths=$paths:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src
paths=$paths:$(pwd)/lib-workflows/torchtunewrapper/src
paths=$paths:$(pwd)/lib-celery/src

PYTHONPATH=$paths python lib-celery/src/pipeline/$1.py "${@:2}"
