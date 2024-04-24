import importlib
import os
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
    job_config_exists = os.path.isfile(os.path.join(
        get_root_path(), job_config_dir, job_config_id + '.py'))
    if not job_config_exists:
        raise FileNotFoundError(f'job_config_id: {job_config_id} not found under {job_config_dir}')

    return importlib.import_module(f'{job_config_dir}.{job_config_id}').job_config


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
