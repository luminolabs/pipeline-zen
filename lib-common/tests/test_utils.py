from logging import Logger

import pytest

from common.utils import load_job_config, get_model_weights_path, get_results_path, \
    get_logs_path, setup_logger


def test_load_job_config():
    # Invalid job config, raise error
    with pytest.raises(FileNotFoundError):
        load_job_config(job_config_name='foo')

    # Valid job config, confirm config is a dict
    config = load_job_config(job_config_name='imdb_nlp_classification')
    assert isinstance(config, dict)


def test_get_results_path():
    # Path is of type `str`
    assert isinstance(get_results_path('test_job_id'), str)


def test_get_model_weights_path():
    # Path is of type `str`
    assert isinstance(get_model_weights_path('test_job_id'), str)


def test_get_logs_path():
    # Path is of type `str`
    assert isinstance(get_logs_path('test_job_id'), str)


def test_setup_logger():
    logger = setup_logger('test_logger', 'test_job_id')
    assert isinstance(logger, Logger)
