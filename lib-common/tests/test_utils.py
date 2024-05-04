from logging import Logger

import pytest

from common.utils import get_root_path, load_job_config, get_model_weights_path, get_results_path, environment, \
    get_logs_path, setup_logger


def test_get_root_path():
    # Path is of type `str`
    assert isinstance(get_root_path(), str)


def test_load_job_config():
    # Invalid job config, raise error
    with pytest.raises(FileNotFoundError):
        load_job_config(job_config_id='foo')

    # Valid job config, confirm config is a dict
    config = load_job_config(job_config_id='imdb_nlp_classification')
    assert isinstance(config, dict)


def test_get_results_path():
    # Path is of type `str`
    assert isinstance(get_results_path(), str)


def test_get_model_weights_path():
    # Path is of type `str`
    assert isinstance(get_model_weights_path(), str)


def test_get_logs_path():
    # Path is of type `str`
    assert isinstance(get_logs_path(), str)


def test_setup_logger():
    logger = setup_logger('test_logger')
    assert isinstance(logger, Logger)


def test_environment():
    # `environment` defaults to `local`
    assert environment == 'local'
