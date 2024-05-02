import importlib
import logging
import os
import sys
from json import JSONEncoder
from typing import Optional
from datetime import datetime

# This affects a few runtime options such as cache and results folders
# Available options so far: `docker`, `local`
environment = os.environ.get("ENVIRONMENT", 'local')


# Job configuration path in relation to repository root path
job_config_dir = 'job_configs'


def get_root_path() -> str:
    """
    Returns the path to a common root between workflows
    This allows workflows to share results, cache, etc
    :return: Root path
    """
    if environment == 'local':
        return os.path.join('..', '..')
    elif environment == 'docker':
        return '.'


def load_job_config(job_config_id: str) -> dict:
    """
    Loads a job config from a job config file

    :param job_config_id: Job config name; this is the file name without the `.py` extension
    :return: Job config dict
    """
    try:
        return importlib.import_module(f'{job_config_dir}.{job_config_id}').job_config
    except ModuleNotFoundError:
        # Raise `FileNotFoundError` as it's more intuitive message to give to the user
        raise FileNotFoundError(f'job_config_id: {job_config_id} not found under {job_config_dir}')


def get_results_path() -> str:
    """
    :return: Returns the path to the results directory
    """
    path = os.path.join(get_root_path(), '.results')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_weights_path(job_id: Optional[str] = None) -> str:
    """
    :param job_id: Job id to use as part of the model weights path
    :return: Returns the path to the model weights file
    """
    path_list = ()
    if job_id:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path_list = (job_id, timestamp + '.pt')
    path = os.path.join(get_results_path(), 'model_weights', *path_list)
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
                 default_log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger

    :param name: The name of the logger
    :param job_id: Job id to use as part of the logger path
    :param default_log_level: The default log level to use, ex. `logging.INFO`
    :return: A logger instance
    """
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_format = logging.Formatter('%(name)s :: %(levelname)s :: %(message)s')

    # Log to stdout and to file
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("{0}/{1}.log".format(get_logs_path(job_id), f'{name}_{timestamp}'))

    # Set the logger format
    stdout_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Configure logger
    pg_logger = logging.getLogger(name)
    pg_logger.setLevel(default_log_level)
    pg_logger.addHandler(stdout_handler)
    pg_logger.addHandler(file_handler)
    return pg_logger


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