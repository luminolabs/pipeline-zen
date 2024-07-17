import os
import platform
from typing import Optional

from celery import Celery, chain

from common.config_manager import config
from common.utils import get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs, get_logs_path
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


@app.task
def torchtunewrapper(_, job_config_name: str, job_id: Optional[str] = None,
                     dataset_id: str = Optional[None], train_file_path: str = None,
                     batch_size: int = 1, shuffle: bool = True, num_epochs: int = 1,
                     use_lora: bool = True,
                     use_single_device: bool = True, num_gpus: int = 1):
    return _torchtunewrapper(
        job_config_name, job_id,
        dataset_id, train_file_path,
        batch_size, shuffle, num_epochs,
        use_lora,
        use_single_device, num_gpus)


@app.task
def upload_results(_, job_id: str):
    """
    Upload results and logs to Google Cloud Storage.
    :param job_id: The job id to associate with the results
    :return:
    """
    # Upload results
    upload_local_directory_to_gcs(get_results_path(job_id), 'lum-pipeline-zen')
    # Upload logs
    upload_local_directory_to_gcs(get_logs_path(job_id), 'lum-pipeline-zen')


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
    Runs the torchtunewrapper workflow and uploads results to cloud storage

    :param args: Arguments passed to the torchtunewrapper function
    :return:
    """
    # Get job id and update it if necessary
    args = list(args)
    job_config_name = args[0]
    job_id = args[1]
    job_id = args[1] = get_or_generate_job_id(job_config_name, job_id)

    # Define workflow tasks
    tasks = [mark_started.s(None, job_id),
             torchtunewrapper.s(*args),
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
    schedule(*torchtunewrapper_parse_args())
    start_worker()
