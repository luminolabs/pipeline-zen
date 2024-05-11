#!/bin/bash

paths=$(pwd)/lib-common/src:$(pwd)/lib-workflows/$1/src

PYTHONPATH=$paths python lib-workflows/$1/src/$1/cli.py "${@:2}"
