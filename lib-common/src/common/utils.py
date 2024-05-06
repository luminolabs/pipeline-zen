import glob
import importlib
import json
import logging
import os
import sys
import uuid
from enum import Enum
from json import JSONEncoder
from os.path import basename
from typing import Optional, Union, List
from datetime import datetime

from google.cloud import storage
from google.cloud.storage import Bucket


class Env(Enum):
    """
    List of all available environments
    """
    PRODUCTION = 'prod'
    DEVELOPMENT = 'dev'
    TESTING = 'test'
    LOCAL = 'local'
    DOCKER = 'docker'
    CELERY = 'celery'


# Job configuration path in relation to repository root path
job_configs_module = 'job_configs'

# Timestamp format to use for logs, results, etc
system_timestamp_format = '%Y-%m-%d-%H-%M-%S'


def get_environment(default: Optional[Env] = None) -> str:
    """
    :return: Returns the environment name
    ex. `local`, `dev`, `prod`, `docker`, `celery`, etc
    Environments can be stacked: ex. `local-celery` or `dev-docker-celery`
    """
    default = default and default.value or Env.LOCAL.value
    return os.environ.get("ENVIRONMENT", default)


def add_environment(environment: Env):
    """
    Stack environment names; ex `local-celery` is a `local` and a `celery` environment
    :param environment: The environment name to add
    :return:
    """
    os.environ['ENVIRONMENT'] = get_environment() + '-' + environment.value


def is_environment(environment: Union[Env, List[Env]]) -> bool:
    """
    Checks if the environment name is an environment
    :param environment: The environment name to check
    :return: Whether the current environment is the one in question
    """
    if isinstance(environment, Env):
        environment = [environment]
    return any(x.value in get_environment() for x in environment)


def get_system_timestamp() -> str:
    """
    :return: A timestamp formatted as a string; use in logs, results, etc
    """
    return datetime.now().strftime(system_timestamp_format)


def get_root_path() -> str:
    """
    Returns the path to a common root between workflows
    This allows workflows to share results, cache, etc
    :return: Root path
    """
    if 'project' in os.getcwd():
        return '../.'
    elif 'pipeline-zen' in os.getcwd():
        return '.'
    else:
        raise EnvironmentError('Please run workflows from the root of the pipeline-zen directory')


def load_job_config(job_config_name: str) -> dict:
    """
    Loads a job config from a job config file

    :param job_config_name: Job config name; this is the file name without the `.py` extension
    :return: Job config dict
    """
    try:
        return importlib.import_module(f'{job_configs_module}.{job_config_name}').job_config
    except ModuleNotFoundError:
        # Raise `FileNotFoundError` as it's more intuitive message to give to the user
        raise FileNotFoundError(f'job_config_id: {job_config_name} not found under {job_configs_module}')


def get_results_path(job_id: str) -> str:
    """
    :return: Returns the path to the results directory
    """
    path = os.path.join(get_root_path(), '.results', job_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_weights_path(job_id: str) -> str:
    """
    :param job_id: Job id to use as part of the model weights path
    :return: Returns the path to the model weights file
    """
    path = os.path.join(get_results_path(job_id),  'weights.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_logs_path(job_id: Optional[str] = None) -> str:
    """
    :return: Returns the path to the logs directory
    """
    path = os.path.join(get_root_path(), '.logs', job_id or '')
    os.makedirs(path, exist_ok=True)
    return path


def setup_logger(name: str, job_id: Optional[str] = None,
                 add_stdout: bool = True,
                 default_log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger

    :param name: The name of the logger
    :param job_id: Job id to use as part of the logger path
    :param add_stdout: Whether to add the stdout logger or not
    :param default_log_level: The default log level to use, ex. `logging.INFO`
    :return: A logger instance
    """
    log_format = logging.Formatter('%(message)s')

    # Log to stdout and to file
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(get_logs_path(job_id), f'{name}.log'))

    # Set the logger format
    stdout_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Configure logger
    pg_logger = logging.getLogger(name)
    pg_logger.setLevel(default_log_level)
    if add_stdout and not is_environment(Env.CELERY):
        pg_logger.addHandler(stdout_handler)
    pg_logger.addHandler(file_handler)
    return pg_logger


def get_or_generate_job_id(job_config_name: str, job_id: Optional[str] = None) -> str:
    """
    Returns the job id if set -
    Otherwise generates a new job id, in this form `<default job id from the job configuration>-<UUID>`

    :param job_config_name: The name of the job configuration
    :param job_id: Job id to use, or if empty, generates a new job id
    :return: The input job id, or the generated job id
    """
    if not job_id:
        job_config = load_job_config(job_config_name)
        job_id = job_config['job_id'] + '-' + str(uuid.uuid4())
    return job_id


def save_job_results(job_id: str, results: dict, job_name: str) -> None:
    """
    Saves the results of a job to a file

    :param job_id: The job id
    :param results: The results to save
    :param job_name: Name of the job
    :return:
    """
    path = os.path.join(get_results_path(job_id), job_name + '.json')
    with open(path, 'w') as f:
        f.write(json.dumps(results))


class AutoJSONEncoder(JSONEncoder):
    """
    A JSON encoder that automatically serializes objects with a `_json()` method,
    such as `Enums` that implement the `_json()` method.
    """
    def default(self, obj):
        try:
            return obj._json()
        except AttributeError:
            return JSONEncoder.default(self, obj)


class JsonEnumBase(Enum):
    """
    Base class for JSON serializable enums.
    """
    def _json(self):
        return str(self.value)


def upload_local_directory_to_gcs(local_path: str, bucket: Union[str, Bucket], gcs_path: Optional[str] = None):
    """
    Upload a local directory to Google Cloud Storage.

    :param local_path: Local path to upload
    :param bucket: Bucket to upload to
    :param gcs_path:
    :return:
    """
    client = storage.Client(project='neat-airport-407301')
    if isinstance(bucket, str):
        bucket = client.get_bucket(bucket)
    gcs_path = gcs_path or basename(local_path)

    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
