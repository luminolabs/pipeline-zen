import os
import platform
import traceback
from typing import Optional

from celery import Celery, chain
from celery.signals import task_failure

from common.agents.system_metrics import SystemSpecs
from common.config_manager import config
from common.gcp import get_results_bucket_name, send_message_to_pubsub, make_gcs_object_public
from common.helpers import heartbeat_wrapper
from common.utils import get_or_generate_job_id, get_results_path, \
    upload_local_directory_to_gcs, get_logs_path, setup_logger, job_meta_context
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
    job_id = kwargs.get('args')[1]  # `job_id` is always the second argument passed to a task
    user_id = kwargs.get('args')[2]  # `user_id` is always the third argument passed to a task
    logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
    # Not raising exception, since it's already raised by the task
    logger.error('Something went wrong during task execution')
    app.control.shutdown()


@app.task
def torchtunewrapper(_, job_id: str, user_id: str, job_config_name: str,
                     dataset_id: Optional[str] = None, train_file_path: str = None,
                     batch_size: int = 1, shuffle: bool = True, num_epochs: int = 1,
                     use_lora: bool = True, use_qlora: bool = False,
                     lr: float = 3e-4, seed: Optional[int] = None,
                     num_gpus: int = 1,
                     pytorch_cuda_alloc_conf: str = None):
    logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
    try:
        return _torchtunewrapper(
            job_id, user_id, job_config_name,
            dataset_id, train_file_path,
            batch_size, shuffle, num_epochs,
            use_lora, use_qlora,
            lr, seed,
            num_gpus,
            pytorch_cuda_alloc_conf)
    except Exception as e:
        # Not raising exception, so that workflow can run `upload_results` task later on
        logger.error(f'`torchtunewrapper` task failed with error: {e}\n{traceback.format_exc()}')
        return False


@app.task
@heartbeat_wrapper("torchtunewrapper", "upload_weights")
def upload_results(mark_finished_result: bool, job_id: str, user_id: str):
    """
    Upload results and logs to Google Cloud Storage.
    :param mark_finished_result: The result of the `mark_finished` task
    :param job_id: The job id to associate with the results
    :param user_id: The user id to associate with the results
    :return:
    """
    # Upload results
    results_path = get_results_path(job_id, user_id)
    results_bucket_name = get_results_bucket_name(config.env_name)
    upload_local_directory_to_gcs(results_path, results_bucket_name)
    # Upload logs
    logs_path = get_logs_path(job_id, user_id)
    upload_local_directory_to_gcs(logs_path, results_bucket_name)

    # If `mark_finished` task failed, we don't do anything else
    if not mark_finished_result:
        return

    # Gather weight files and other files
    weight_files = [f for f in os.listdir(results_path) if f.endswith('.pt')]
    other_files = [f for f in os.listdir(results_path) if f in ['config.json']]
    # Make files public in GCS
    for f in weight_files + other_files:
        make_gcs_object_public(results_bucket_name, f'{user_id}/{job_id}/{f}')


@app.task
def delete_results(_, job_id: str, user_id: str):
    """
    Deletes the job results and logs directories.

    :param job_id: The job id to delete
    :param user_id: The user id to delete
    :return:
    """
    results_path = get_results_path(job_id, user_id)
    logs_path = get_logs_path(job_id, user_id)
    os.system(f'rm -rf {results_path}')
    os.system(f'rm -rf {logs_path}')


@app.task
def mark_finished(torchtunewrapper_result, job_id: str, user_id: str):
    """
    Creates a `.finished` file that serves as a signal to listeners
    that the job finished.

    :param torchtunewrapper_result: The result of the torchtunewrapper task
    :param job_id: The job id that finished
    :param user_id: The user id that finished the job
    :return:
    """
    logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
    if not torchtunewrapper_result:
        # Not touching this file allows the startup script to mark job as failed
        logger.warning(f'`torchtunewrapper` task failed - will not run `mark_finished` task')
        return False

    # Write job metadata to file and publish to Pub/Sub
    results_path = get_results_path(job_id, user_id)
    results_bucket_name = get_results_bucket_name(config.env_name)
    weight_files = [f for f in os.listdir(results_path) if f.endswith('.pt')]
    other_files = [f for f in os.listdir(results_path) if f in ['config.json']]
    weights_data = {
        'action': 'job_artifacts',
        'workflow': 'torchtunewrapper',
        'base_url': f'https://storage.googleapis.com/'
                    f'{results_bucket_name}/{user_id}/{job_id}',
        'weight_files': weight_files,
        'other_files': other_files
    }
    # Send message to Pub/Sub
    send_message_to_pubsub(job_id, user_id, config.jobs_meta_topic, weights_data)
    # Write job metadata to file
    with job_meta_context(job_id, user_id) as job_meta:
        del weights_data['action']
        job_meta['weights'] = weights_data

    path = os.path.join(config.root_path, config.results_path, config.finished_file)
    with open(path, "w") as f:
        f.write(job_id + "\n")
    return True


@app.task
def mark_started(_, job_id: str, user_id: str):
    """
    Creates a `.started` file that serves as a signal to listeners
    that the job started.

    :param job_id: The job id that started
    :param user_id: The user id that started the job
    :return:
    """
    path = os.path.join(config.root_path, config.results_path, config.started_file)
    with open(path, "w") as f:
        f.write(job_id + "\n")


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
    Runs the torchtunewrapper workflow and uploads results to cloud storage

    :param args: Arguments passed to the torchtunewrapper function
    :return:
    """
    # Get job id and update it if necessary
    args = list(args)
    job_config_name = args[2]
    job_id = args[0]
    job_id = args[0] = get_or_generate_job_id(job_config_name, job_id)
    user_id = args[1]

    # On non-local environments, we require the presence of GPUs
    if config.env_name != 'local':
        logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
        system_specs = SystemSpecs(logger)
        if system_specs.get_gpu_spec() is None:
            raise RuntimeError('No GPUs found on this machine')

    # Define workflow tasks
    tasks = [mark_started.s(None, job_id, user_id),
             torchtunewrapper.s(*args),
             mark_finished.s(job_id, user_id)]
    # Add task to upload job results (when not on a local or test environment)
    if config.upload_results:
        tasks.append(upload_results.s(job_id, user_id))
    # Add task to delete job results if in non-ephemeral environment
    if config.delete_results:
        tasks.append(delete_results.s(job_id, user_id))
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
    schedule(*torchtunewrapper_parse_args())
    start_worker()
