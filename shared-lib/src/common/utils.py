import importlib
import os
from typing import Optional
from datetime import datetime

# This affects a few runtime options such as cache and results folders
# Available options so far: `docker`, `local`
environment = os.environ.get("ENVIRONMENT", 'local')


# Returns the path to a common root between workflows
# This allows workflows to share results, cache, etc
def get_root_path() -> str:
    if environment == 'local':
        return os.path.join('..', '..')
    elif environment == 'docker':
        return '.'


def load_job_config(job_config_id: str) -> dict:
    return importlib.import_module(f'job_configs.{job_config_id}').job_config


def get_results_path() -> str:
    path = os.path.join(get_root_path(), '.results')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_weights_path(job_id: Optional[str] = None) -> str:
    path_list = ()
    if job_id:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path_list = (job_id, timestamp + '.pt')
    path = os.path.join(get_results_path(), 'model_weights', *path_list)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
