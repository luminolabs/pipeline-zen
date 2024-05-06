import os
import platform

from celery import Celery, chain

import celeryconfig
from common.utils import add_environment, Env
from train.cli import parse_args as train_parse_args
from train.workflow import main as _train
from evaluate.workflow import main as _evaluate

# OSX compatibility
if platform.system() == 'Darwin':
    # Hugging Face library throws some warnings when run within
    # the Celery environment, probably because both libraries
    # are running parallelization internally, and OSX doesn't
    # seem to like this
    os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Update environment name
add_environment(Env.CELERY)

# Setup Celery App
app = Celery('train_evaluate')
app.config_from_object(celeryconfig)


@app.task
def train(job_config_name: str, job_id: str, batch_size: int, num_epochs: int, num_batches: int):
    model_weights_path = _train(job_config_name, job_id, batch_size, num_epochs, num_batches)
    return model_weights_path


@app.task
def evaluate(model_weights: str, job_config_name: str, job_id: str, batch_size: int, num_batches: int):
    return _evaluate(job_config_name, model_weights, job_id, batch_size, num_batches)


def schedule(*args):
    """
    Runs the train and evaluate workflows one after the other
    :param args: Arguments passed to the train and evaluate functions
    :return:
    """
    job_config_name, job_id, batch_size, num_epochs, num_batches = args
    train_args = (job_config_name, job_id, batch_size, num_epochs, num_batches)
    evaluate_args = (job_config_name, job_id, batch_size, num_batches)

    # Output from `train` automatically goes into `evaluate` method's first argument,
    # which in this case is the relative path to the trained weights.
    chain(train.s(*train_args), evaluate.s(*evaluate_args))()


def start_worker():
    # Start the celery worker
    # NOTE: The worker will continue running after the task queue is processed
    argv = [
        'worker',
        '--loglevel=INFO',
        '--pool=solo'
    ]
    app.worker_main(argv)


if __name__ == '__main__':
    schedule(*train_parse_args())
    start_worker()
