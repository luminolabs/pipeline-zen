import os
import platform

from celery import Celery, chain

import celeryconfig
from common.utils import add_environment, Env, get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs, is_environment
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
def train(_, job_config_name: str, job_id: str, batch_size: int, num_epochs: int, num_batches: int):
    return _train(job_config_name, job_id, batch_size, num_epochs, num_batches)


@app.task
def evaluate(_, job_config_name: str, job_id: str, batch_size: int, num_batches: int):
    return _evaluate(job_config_name, job_id, batch_size, num_batches)


@app.task
def upload_results(_, job_id: str):
    """
    Upload results to Google Cloud Storage.
    :return:
    """
    upload_local_directory_to_gcs(get_results_path(job_id), 'lum-pipeline-zen')


def schedule(*args):
    """
    Runs the train and evaluate workflows one after the other

    :param args: Arguments passed to the train and evaluate functions
    :return:
    """
    job_config_name, job_id, batch_size, num_epochs, num_batches = args
    job_id = get_or_generate_job_id(job_config_name, job_id)

    train_args = (job_config_name, job_id, batch_size, num_epochs, num_batches)
    evaluate_args = (job_config_name, job_id, batch_size, num_batches)

    # Define workflow `train` -> `evaluate`
    tasks = [train.s(None, *train_args), evaluate.s(*evaluate_args)]
    # Upload job results when not on a local or test environment
    if not is_environment([Env.LOCAL, Env.TESTING]):
        tasks.append(upload_results.s(job_id))
    # Schedule tasks
    chain(*tasks)()


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
