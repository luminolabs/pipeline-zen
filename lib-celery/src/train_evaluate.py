import os
import platform
import uuid

from celery import Celery, chain

import celeryconfig
from common.utils import add_environment
from train import main as _train
from evaluate import main as _evaluate

# OSX compatibility
if platform.system() == 'Darwin':
    # Hugging Face library throws some warnings when run within
    # the Celery environment, probably because both libraries
    # are running parallelization internally, and OSX doesn't
    # seem to like this
    os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Update environment name
add_environment('celery')

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


def test():
    """
    Testing function; runs the train and evaluate workflows one after the other
    :return:
    """
    job_config_name = 'agnews_nlp_classification'
    job_id = job_config_name + '-' + str(uuid.uuid4())
    batch_size = 8
    num_epochs = 2
    num_batches = 3
    args1 = (job_config_name, job_id, batch_size, num_epochs, num_batches)
    args2 = (job_config_name, job_id, batch_size, num_batches)

    # Output from `train` automatically goes into `evaluate` method's first argument,
    # which in this case is the relative path to the trained weights.
    chain(train.s(*args1), evaluate.s(*args2))()

    # Start the celery worker
    # NOTE: The worker has to be stopped manually when the job above is finished
    argv = [
        'worker',
        '--loglevel=INFO',
        '--pool=solo'
    ]
    app.worker_main(argv)


if __name__ == '__main__':
    test()
