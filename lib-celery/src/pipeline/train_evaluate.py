import os
import platform

from celery import Celery, chain

from common.config_manager import config
from common.utils import get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs
from train.cli import parse_args as train_parse_args
from train.workflow import main as _train
from evaluate.workflow import main as _evaluate

# Celery logs output to stdout
# Disable workflow logger output to stdout
# so that logs aren't logged twice
config.log_stdout = False

# OSX compatibility
if platform.system() == 'Darwin':
    # Hugging Face library throws some warnings when run within
    # the Celery environment, probably because both libraries
    # are running parallelization internally, and OSX doesn't
    # seem to like this
    os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Setup Celery App
app = Celery('train_evaluate', broker=config.celery_broker_url)


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


@app.task
def mark_finished(_, job_id: str):
    path = os.path.join(config.root_path, config.finished_file, job_id)
    with open(path, "w") as f:
        f.write(f'job_id: {job_id}')


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

    # Define workflow tasks: `train` -> `evaluate`
    tasks = [train.s(None, *train_args), evaluate.s(*evaluate_args)]
    # Add task to upload job results (when not on a local or test environment)
    if config.upload_results:
        tasks.append(upload_results.s(job_id))
    # If we're not local, then create the `.finished` file
    # that's used by the deployment script to watch for job
    # completion
    if config.env_name != 'local':
        tasks.append(mark_finished.s(job_id))
    # Send task chain to celery scheduler
    chain(*tasks)()


def start_worker():
    """
    Starts the celery worker
    NOTE: The worker will continue running after the task queue is processed
    :return:
    """
    argv = [
        'worker',
        '--loglevel=INFO',
        '--pool=solo'
    ]
    app.worker_main(argv)


if __name__ == '__main__':
    schedule(*train_parse_args())
    start_worker()
