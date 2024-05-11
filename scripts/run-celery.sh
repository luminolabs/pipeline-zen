paths=$(pwd)/lib-common/src:$(pwd)/lib-workflows/train/src:$(pwd)/lib-workflows/evaluate/src:$(pwd)/lib-celery/src

PYTHONPATH=$paths python lib-celery/src/pipeline/train_evaluate.py "${@}"
