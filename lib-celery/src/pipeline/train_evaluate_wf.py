import os
import platform

from celery import Celery, chain

from common.config_manager import config
from common.gcp import get_results_bucket_name
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
    :param job_id: The job id to associate with the results
    :return:
    """
    results_bucket_name = get_results_bucket_name(config.env_name)
    upload_local_directory_to_gcs(get_results_path(job_id), results_bucket_name)


@app.task
def mark_finished(_, job_id: str):
    """
    Creates a `.finished` file that serves as a signal to listeners
    that the job finished.

    :param job_id: The job id that finished
    :return:
    """
    path = os.path.join(config.root_path, config.results_path, config.finished_file)
    with open(path, "w") as f:
        f.write(job_id)


@app.task
def mark_started(_, job_id: str):
    """
    Creates a `.started` file that serves as a signal to listeners
    that the job started.

    :param job_id: The job id that started
    :return:
    """
    path = os.path.join(config.root_path, config.results_path, config.started_file)
    with open(path, "w") as f:
        f.write(job_id)


@app.task
def shutdown_celery_worker(_):
    """
    Shuts down the celery worker.
    """
    # sends shutdown signal to *all( workers
    # ...there's just one worker though,
    # because we aren't using a distributed queue yet
    app.control.shutdown()


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
    tasks = [mark_started.s(None, job_id),
             train.s(*train_args), evaluate.s(*evaluate_args),
             mark_finished.s(job_id)]
    # Add task to upload job results (when not on a local or test environment)
    if config.upload_results:
        tasks.append(upload_results.s(job_id))
    # Shut down worker, since we aren't using a
    # distributed job queue yet in any environment
    tasks.append(shutdown_celery_worker.s())
    # Send task chain to celery scheduler
    chain(*tasks)()


def start_worker():
    """
    Starts the celery worker
    NOTE: The worker will continue running after the task queue is processed
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
