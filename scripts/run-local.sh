paths=$(pwd):$(pwd)/lib-common/src:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src

PYTHONPATH=$paths python lib-workflows/$1/src/cli.py "${@:2}"
