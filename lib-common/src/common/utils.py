import glob
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from json import JSONEncoder
from os.path import basename
from typing import Optional, Union

from google.cloud import storage
from google.cloud.storage import Bucket
from omegaconf import OmegaConf, DictConfig

from common.config_manager import config

# Timestamp format to use for logs, results, etc
system_timestamp_format = '%Y-%m-%d-%H-%M-%S'


class JsonEnumBase(Enum):
    """
    Base class for JSON serializable enums.
    """

    def _json(self):
        return str(self.value)


class JobCategory(JsonEnumBase):
    NLP = 'nlp'
    IMAGE = 'image'
    LLM = 'llm'


class JobType(JsonEnumBase):
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    INSTRUCTION = 'instruction'


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
            # If the object has a `_json()` method, use it;
            # it means it's an object (ex. Enum) that implements the `_json()` method
            return obj._json()
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


def get_results_path(job_id: str = '') -> str:
    """
    :param job_id: Job id to use as part of the results path, can be left empty to get the root results folder
    :return: Returns the path to the results directory
    """
    path = os.path.join(config.root_path, config.results_path, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_model_weights_path(job_id: str) -> str:
    """
    :param job_id: Job id to use as part of the model weights path
    :return: Returns the path to the model weights file
    """
    path = os.path.join(get_results_path(job_id), config.weights_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_logs_path(job_id: str) -> str:
    """
    :param job_id: Job id to use as part of the logs path
    :return: Returns the path to the logs directory
    """
    path = os.path.join(config.root_path, config.logs_path, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def setup_logger(name: str, job_id: str, user_id: str,
                 add_stdout: bool = True,
                 log_level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger

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
        f'{config.env_name} - %(name)s - %(levelname)s - %(asctime)s - %(message)s - job_id: {job_id} - user_id: {user_id}',
        datefmt=system_timestamp_format
    )
    # Set the logger format to use the current UTC time
    log_format.converter = lambda *args: utcnow().timetuple()

    # Log to stdout and to file
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(get_logs_path(job_id), f'{name}.log'))

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


def upload_local_directory_to_gcs(local_path: str, bucket: Union[str, Bucket], gcs_path: Optional[str] = None):
    """
    Upload a local directory to Google Cloud Storage.

    :param local_path: Local path to upload
    :param bucket: Bucket to upload to
    :param gcs_path:
    :return:
    """
    client = storage.Client(project=config.gcp_project)
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
    if job_config['category'] == JobCategory.LLM:
        return DictConfig({**job_config})

    # Construct the template configuration file; ex. `image_segmentation_base.yml`
    template_config_name = (job_config['category'].value +
                            '_' + job_config['type'].value +
                            '_base.yml')

    # Pull the base and template configuration files
    base_config = read_job_config_from_file(os.path.join('templates', 'base.yml'))
    template_config = read_job_config_from_file(os.path.join('templates', template_config_name))

    # Build final configuration dict
    return DictConfig({**base_config, **template_config, **job_config})


def read_job_config_from_file(job_config_name: str,
                              overrides: Optional[dict] = None, is_torchtune: bool = False) -> DictConfig:
    """
    Reads the job config from a YAML file

    :param job_config_name: Name of the job configuration
    :param overrides: Overrides to apply to the job config
    :param is_torchtune: Whether the job config is a torchtune config file or not
    :return: The job config dictionary
    """
    # Normalize filename
    if not job_config_name.endswith('.yml'):
        job_config_name += '.yml'

    # Open and read YAML into dictionary
    path = os.path.join(
        config.root_path, config.job_configs_path,
        'torchtune' if is_torchtune else '',
        job_config_name)
    try:
        job_config = OmegaConf.merge(OmegaConf.load(path), overrides or {})
    except FileNotFoundError:
        # User friendly error
        raise FileNotFoundError(f'job_config_name: {job_config_name} not found under {path}')

    if not is_torchtune:
        # Resolve these two into the Enum class; ex. `IMAGE` -> `JobCategory.IMAGE`
        if job_config.get('type'):
            job_config['type'] = getattr(JobType, job_config['type'])
        if job_config.get('category'):
            job_config['category'] = getattr(JobCategory, job_config['category'])

    return job_config


def utcnow() -> datetime:
    """
    :return: The current UTC time
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def utcnow_str() -> str:
    """
    :return: The current UTC time as a string
    """
    return utcnow().strftime(system_timestamp_format)
