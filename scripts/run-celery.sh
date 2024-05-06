paths=$(pwd):$(pwd)/lib-common/src:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src:$(pwd)/lib-celery/src

PYTHONPATH=$paths python lib-celery/src/train_evaluate.py "${@}"
