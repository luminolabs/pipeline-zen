import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from json import JSONEncoder
from typing import Optional

from filelock import FileLock
from omegaconf import OmegaConf, DictConfig

from common.config_manager import config

#################
### CONSTANTS ###
#################


# insert constants here
# ...


##################
### TIME UTILS ###
##################


# Timestamp format to use for logs, results, etc
system_timestamp_format = '%Y-%m-%dT%H:%M:%SZ'


def utcnow() -> datetime:
    """
    :return: The current UTC time
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def utcnow_str(fmt: str = system_timestamp_format) -> str:
    """
    :param fmt: The format to use for the timestamp
    :return: The current UTC time as a string
    """
    return utcnow().strftime(fmt)


#####################
### LOGGING UTILS ###
#####################


class JsonEnumBase(Enum):
    """
    Base class for JSON serializable enums.
    """

    def json(self):
        return str(self.value)


class AutoJSONEncoder(JSONEncoder):
    """
    A JSON encoder that automatically serializes objects with a `_json()` method,
    such as `Enums` that implement the `_json()` method.
    """
    def default(self, obj):
        # Convert OmegaConf objects to dictionaries
        # before serializing them to JSON strings to avoid errors
        if isinstance(obj, DictConfig):
            obj = dict(obj)
        try:
            # If the object has a `json()` method, use it;
            # it means it's an object (ex. Enum) that implements the `json()` method
            return obj.json()
        except AttributeError:
            # Otherwise, return the JSON serialized object
            return json.dumps(obj)


class JsonFormatter(logging.Formatter):
    def __init__(self, job_id: str, user_id: str):
        """
        :param job_id: The job id to use in the log record
        :param user_id: The user id to use in the log record
        """
        super(JsonFormatter, self).__init__()
        # Set the JSON encoder to use for the log records
        self.json_encoder = AutoJSONEncoder()
        self.job_id = job_id
        self.user_id = user_id

    def format(self, record) -> str:
        """
        Formats the log record as a JSON string
        :param record: The log record to format
        :return: The JSON formatted log record
        """
        # Define the log record format
        log_record = {
            "env": config.env_name,
            "logger_name": record.name,
            "level": record.levelname,
            "timestamp": utcnow_str(),
            "message": record.getMessage(),
            "job_id": self.job_id,
            "user_id": self.user_id
        }
        # Add any extra attributes
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        # Return the JSON formatted log record
        return json.dumps(log_record)


def setup_logger(name: str, job_id: str, user_id: str,
                 add_stdout: bool = True,
                 log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up the logger

    :param name: The name of the logger
    :param job_id: Job id to use as part of the logger path
    :param user_id: User id to use as part of the logger path
    :param add_stdout: Whether to add the stdout logger or not
    :param log_level: The log level to log at, ex. `logging.INFO`
    :return: A logger instance
    """
    log_level = log_level or config.log_level

    # Set the logger format for stdout
    log_format = logging.Formatter(
        f'{config.env_name} - %(name)s - %(levelname)s - %(asctime)s - %(message)s - '
        f'job_id: {job_id} - user_id: {user_id}',
        datefmt=system_timestamp_format
    )
    # Set the logger format to use the current UTC time
    log_format.converter = lambda *args: utcnow().timetuple()
    # Log to stdout and to file
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(get_work_dir(job_id, user_id), f'{name}.log'))
    # Set the logger formats
    stdout_handler.setFormatter(log_format)
    file_handler.setFormatter(JsonFormatter(job_id, user_id))
    # Configure logger
    pg_logger = logging.getLogger(name)
    pg_logger.setLevel(log_level)
    if add_stdout and config.log_stdout:
        pg_logger.addHandler(stdout_handler)
    pg_logger.addHandler(file_handler)
    return pg_logger


##################
### FILE UTILS ###
##################


def get_work_dir(job_id: str, user_id: str) -> str:
    """
    :return: Returns the path to the results directory for a given job and user
    """
    path = os.path.join(config.root_path, config.work_dir, user_id, job_id)
    os.makedirs(path, exist_ok=True)
    return str(path)


def _read_job_meta_from_file(path: str) -> dict:
    """
    :return: Returns the job meta dictionary for a given job and user
    """

    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def _write_job_meta_to_file(path: str, job_meta: dict) -> None:
    """
    :return: Writes the job meta dictionary to a file for a given job and user
    """
    with open(path, 'w') as f:
        json.dump(job_meta, f, indent=4)


@contextmanager
def job_meta_context(job_id: str, user_id: str):
    """
    This context manager fully manages reading and writing the job metadata to a file

    Usage:
        # file contents are loaded into job_meta, on `with` block entry
        with job_meta_context(job_id, user_id) as job_meta:
            job_meta['key'] = 'value'
        # job_meta is written back to the file, on `with` block exit
    """
    path = os.path.join(get_work_dir(job_id, user_id), config.job_meta_file)
    with FileLock(path + '.lock', thread_local=False):
        job_meta = _read_job_meta_from_file(path)
        yield job_meta
        _write_job_meta_to_file(path, job_meta)


#################
### JOB UTILS ###
#################


def save_job_results(job_id: str, user_id: str, results: dict, job_name: str) -> None:
    """
    Saves the results of a job to a file

    :param job_id: The job id
    :param user_id: The user id
    :param results: The results to save
    :param job_name: Name of the job
    :return:
    """
    path = os.path.join(get_work_dir(job_id, user_id), job_name + '.json')
    with open(path, 'w') as f:
        f.write(json.dumps(results))


####################
### CONFIG UTILS ###
####################


def is_local_env() -> bool:
    """
    :return: True if the environment is local, False otherwise
    """
    return config.env_name == config.local_env_name


def load_job_config(job_config_name: str) -> DictConfig:
    """
    Builds the job configuration dict using the requested config plus values
    inherited from the default templates

    :param job_config_name: Name of the job configuration
    :return: Final job configuration dict
    """
    # Pull the requested configuration file
    job_config = read_job_config_from_file(job_config_name)
    # If the job category is LLM, return the job config as is, no need for other modifications
    return DictConfig({**job_config})


def read_job_config_from_file(job_config_name: str,
                              overrides: Optional[dict] = None, sub_dir: Optional[str] = '') -> DictConfig:
    """
    Reads the job config from a YAML file

    :param job_config_name: Name of the job configuration
    :param overrides: Overrides to apply to the job config
    :param sub_dir: Sub-directory to look for the job config
    :return: The job config dictionary
    """
    # Normalize filename
    if not job_config_name.endswith('.yml'):
        job_config_name += '.yml'
    # Open and read YAML
    path = str(os.path.join(config.root_path, config.job_configs_path, sub_dir, job_config_name))
    try:
        job_config = OmegaConf.merge(OmegaConf.load(path), overrides or {})
        return job_config
    except FileNotFoundError:
        # User friendly error
        raise FileNotFoundError(f'job_config_name: {job_config_name} not found under {path}')
