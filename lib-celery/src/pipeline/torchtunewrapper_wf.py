import os
import platform
import traceback
from typing import Optional

from celery import Celery, chain
from celery.signals import task_failure

from common.config_manager import config
from common.gcp import get_results_bucket_name
from common.utils import get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs, get_logs_path, setup_logger
from torchtunewrapper.cli import parse_args as torchtunewrapper_parse_args
from torchtunewrapper.workflow import main as _torchtunewrapper

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
app = Celery('torchwrapper', broker=config.celery_broker_url)


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
    # `job_id` is always the second argument passed to a task
    job_id = kwargs.get('args')[1]
    logger = setup_logger('celery_torchtunewrapper_wf', job_id)
    # Not raising exception, since it's already raised by the task
    logger.error('Something went wrong during task execution')
    app.control.shutdown()


@app.task
def torchtunewrapper(_, job_id: str, job_config_name: str,
                     dataset_id: str = Optional[None], train_file_path: str = None,
                     batch_size: int = 1, shuffle: bool = True, num_epochs: int = 1,
                     use_lora: bool = True, use_qlora: bool = False,
                     num_gpus: int = 1,
                     pytorch_cuda_alloc_conf: str = None):
    logger = setup_logger('celery_torchtunewrapper_wf', job_id)
    try:
        return _torchtunewrapper(
            job_id, job_config_name,
            dataset_id, train_file_path,
            batch_size, shuffle, num_epochs,
            use_lora, use_qlora,
            num_gpus,
            pytorch_cuda_alloc_conf)
    except Exception as e:
        # Not raising exception, so that workflow can run `upload_results` task later on
        logger.error(f'`torchtunewrapper` task failed with error: {e}\n{traceback.format_exc()}')
        return None


@app.task
def upload_results(_, job_id: str):
    """
    Upload results and logs to Google Cloud Storage.
    :param job_id: The job id to associate with the results
    :return:
    """
    results_bucket_name = get_results_bucket_name(config.env_name)
    # Upload results
    upload_local_directory_to_gcs(get_results_path(job_id), results_bucket_name)
    # Upload logs
    upload_local_directory_to_gcs(get_logs_path(job_id), results_bucket_name)


@app.task
def mark_finished(torchtunewrapper_result, job_id: str):
    """
    Creates a `.finished` file that serves as a signal to listeners
    that the job finished.

    :param torchtunewrapper_result: The result of the torchtunewrapper task
    :param job_id: The job id that finished
    :return:
    """
    logger = setup_logger('celery_torchtunewrapper_wf', job_id)
    if not torchtunewrapper_result:
        # Not touching this file allows the startup script to mark job as failed
        logger.warning(f'`torchtunewrapper` task failed - will not run `mark_finished` task')
        return None
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
def shutdown_celery_worker(_, job_id: str):
    """
    Shuts down the celery worker.
    """
    # sends shutdown signal to *all workers*
    # ...there's just one worker though,
    # because we aren't using a distributed queue yet
    app.control.shutdown()


def schedule(*args):
    """
    Runs the torchtunewrapper workflow and uploads results to cloud storage

    :param args: Arguments passed to the torchtunewrapper function
    :return:
    """
    # Get job id and update it if necessary
    args = list(args)
    job_config_name = args[1]
    job_id = args[0]
    job_id = args[0] = get_or_generate_job_id(job_config_name, job_id)

    # Define workflow tasks
    tasks = [mark_started.s(None, job_id),
             torchtunewrapper.s(*args),
             mark_finished.s(job_id)]
    # Add task to upload job results (when not on a local or test environment)
    if config.upload_results:
        tasks.append(upload_results.s(job_id))
    # Shut down worker, since we aren't using a
    # distributed job queue yet in any environment
    tasks.append(shutdown_celery_worker.s(job_id))
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
    schedule(*torchtunewrapper_parse_args())
    start_worker()
