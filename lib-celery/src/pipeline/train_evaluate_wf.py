import os
import platform
import traceback

from celery import Celery, chain
from celery.signals import task_failure

from common.config_manager import config
from common.gcp import get_results_bucket_name
from common.helpers import heartbeat_wrapper
from common.utils import get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs, setup_logger
from evaluate.workflow import main as _evaluate
from train.cli import parse_args as train_parse_args
from train.workflow import main as _train

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


@task_failure.connect
def handle_task_failure(*args, **kwargs):
    """
    Handles task failures.

    `train` and `evaluate` tasks have special logic that allows the workflow to continue without
    reaching this function. We do this so that if `train` or `evaluate` fail, we can still run
    the `upload_results` task, which will upload whatever artifacts and logs were generated before
    the error occurred.

    If any other task fails, we terminate the workflow as well as the worker, so that the whole
    script execution ends and the worker VM can shut down.
    """
    # `job_id` and `user_id` are passed as arguments to the task
    job_id = kwargs.get('args')[1]
    user_id = kwargs.get('args')[2]
    logger = setup_logger('celery_train_evaluate_wf', job_id, user_id)
    # Not raising exception, since it's already raised by the task
    logger.error('Something went wrong during task execution')
    app.control.shutdown()


@app.task
@heartbeat_wrapper("train_evaluate", "train")
def train(_, job_id: str, user_id: str, job_config_name: str, batch_size: int, num_epochs: int, num_batches: int):
    logger = setup_logger('celery_train_evaluate_wf', job_id, user_id)
    try:
        return _train(job_id, user_id, job_config_name, batch_size, num_epochs, num_batches)
    except Exception as e:
        # Not raising exception, so that workflow can run `upload_results` task later on
        logger.error(f'`train` task failed with error: {e}\n{traceback.format_exc()}')
        return False


@app.task
@heartbeat_wrapper("train_evaluate", "evaluate")
def evaluate(train_result, job_id: str, user_id: str, job_config_name: str, batch_size: int, num_batches: int):
    logger = setup_logger('celery_train_evaluate_wf', job_id, user_id)
    if not train_result:
        logger.warning(f'`train` task failed - will not run `evaluate` task')
        return False
    try:
        return _evaluate(job_id, user_id, job_config_name, batch_size, num_batches)
    except Exception as e:
        # Not raising exception, so that workflow can run `upload_results` task later on
        logger.error(f'`evaluate` task failed with error: {e}\n{traceback.format_exc()}')
        return False


@app.task
@heartbeat_wrapper("train_evaluate", "upload_weights")
def upload_results(_, job_id: str, user_id: str):
    """
    Upload results to Google Cloud Storage.
    :param job_id: The job id to associate with the results
    :param user_id: The user id to associate with the results
    :return:
    """
    results_bucket_name = get_results_bucket_name(config.env_name)
    upload_local_directory_to_gcs(get_results_path(job_id, user_id), results_bucket_name)


@app.task
def mark_finished(evaluate_result, job_id: str, user_id: str):
    """
    Creates a `.finished` file that serves as a signal to listeners
    that the job finished.

    :param evaluate_result: Result of the evaluation task
    :param job_id: The job id that finished
    :param user_id: The user id that started the job
    :return:
    """
    logger = setup_logger('celery_train_evaluate_wf', job_id, user_id)
    if not evaluate_result:
        # Not touching this file allows the startup script to mark job as failed
        logger.warning(f'`evaluate` task failed - will not run `mark_finished` task')
        return False
    path = os.path.join(config.root_path, get_results_path(), config.finished_file)
    with open(path, "w") as f:
        f.write(job_id)


@app.task
def mark_started(_, job_id: str, user_id: str):
    """
    Creates a `.started` file that serves as a signal to listeners
    that the job started.

    :param job_id: The job id that started
    :param user_id: The user id that started the job
    :return:
    """
    path = os.path.join(config.root_path, get_results_path(), config.started_file)
    with open(path, "w") as f:
        f.write(job_id)


@app.task
def shutdown_celery_worker(_, job_id: str, user_id: str):
    """
    Shuts down the celery worker.
    """
    # sends shutdown signal to *all workers*
    # ...there's just one worker though,
    # because we aren't using a distributed queue yet
    app.control.shutdown()


def schedule(*args):
    """
    Runs the train and evaluate workflows one after the other

    :param args: Arguments passed to the train and evaluate functions
    :return:
    """
    job_id, user_id, job_config_name, batch_size, num_epochs, num_batches = args
    job_id = get_or_generate_job_id(job_config_name, job_id)

    train_args = (job_id, user_id, job_config_name, batch_size, num_epochs, num_batches)
    evaluate_args = (job_id, user_id, job_config_name, batch_size, num_batches)

    # Define workflow tasks: `train` -> `evaluate`
    tasks = [mark_started.s(None, job_id, user_id),
             train.s(*train_args), evaluate.s(*evaluate_args),
             mark_finished.s(job_id, user_id)]
    # Add task to upload job results (when not on a local or test environment)
    if config.upload_results:
        tasks.append(upload_results.s(job_id, user_id))
    # Shut down worker, since we aren't using a
    # distributed job queue yet in any environment
    tasks.append(shutdown_celery_worker.s(job_id, user_id))
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
