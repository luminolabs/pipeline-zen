import json
import logging
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

import pytest

from common.agent.job_logger import TorchtunewrapperLoggerAgent
from common.agent.system_specs import SystemSpecsAgent


@pytest.fixture
def mock_config():
    with patch('common.agent.job_logger.config') as mock_cfg:
        # Configure the mock
        mock_cfg.gcp_project = 'test-project'
        mock_cfg.bq_dataset = 'test_dataset'
        mock_cfg.jobs_meta_topic = 'test-topic'
        mock_cfg.send_to_bq = True
        mock_cfg.send_to_pubsub = True
        yield mock_cfg


@pytest.fixture
def mock_loggers():
    agent_logger = MagicMock(spec=logging.Logger)
    main_logger = MagicMock(spec=logging.Logger)
    return agent_logger, main_logger


@pytest.fixture
def logger_agent(mock_config, mock_loggers):
    agent_logger, main_logger = mock_loggers
    return TorchtunewrapperLoggerAgent('test-job', 'test-user', agent_logger, main_logger)


def test_mark_time_start(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers

    logger_agent.mark_time_start()

    assert logger_agent._time_start is not None
    assert isinstance(logger_agent._time_start, datetime)
    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'time_start'
    assert logged_data['job_id'] == 'test-job'
    assert logged_data['user_id'] == 'test-user'


def test_mark_time_end(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers

    logger_agent.mark_time_end()

    assert logger_agent._time_end is not None
    assert isinstance(logger_agent._time_end, datetime)
    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'time_end'


def test_log_time_elapsed(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers
    logger_agent.mark_time_start()
    logger_agent.mark_time_end()

    logger_agent.log_time_elapsed()

    agent_logger.info.assert_called()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'time_elapsed'
    assert 'minutes' in logged_data['data']


@patch('common.agent.job_logger.SystemSpecsAgent')
def test_log_system_specs(mock_system_specs_cls, logger_agent, mock_loggers):
    agent_logger, main_logger = mock_loggers
    mock_specs = {'cpu': 'test-cpu', 'gpu': 'test-gpu', 'mem': '16GB'}
    mock_system_specs = MagicMock(spec=SystemSpecsAgent)
    mock_system_specs.get_specs.return_value = mock_specs
    mock_system_specs_cls.return_value = mock_system_specs

    logger_agent.log_system_specs()

    mock_system_specs_cls.assert_called_once_with(main_logger)
    mock_system_specs.get_specs.assert_called_once()
    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'system_specs'
    assert logged_data['data'] == mock_specs


def test_log_job_config(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers
    test_config = {'param1': 'value1', 'param2': 'value2'}

    logger_agent.log_job_config(test_config)

    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'job_config'
    assert logged_data['data'] == test_config


@patch('common.agent.job_logger.insert_to_biqquery')
def test_bq_insert(mock_insert_bq, logger_agent, mock_loggers, mock_config):
    test_data = {'key': 'value', 'data': 'test'}
    logger_agent._bq_insert(test_data)

    mock_insert_bq.assert_called_once()
    args = mock_insert_bq.call_args[0]
    assert 'test-project.test_dataset.torchtunewrapper' in args
    assert 'key' in args[1]
    assert 'test' == json.loads(args[1]['data'])['value']


@patch('common.agent.job_logger.publish_to_pubsub')
def test_pubsub_send(mock_publish, logger_agent):
    test_row = {'operation': 'test', 'data': {'key': 'value'}}

    logger_agent._pubsub_send(test_row)

    mock_publish.assert_called_once_with(
        'test-job', 'test-user',
        topic_name='test-topic',
        message={'sender': 'job_logger', **test_row}
    )


@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_file_write(mock_json_dump, mock_file_open, logger_agent):
    test_row = {'operation': 'test', 'data': 'test-data'}

    with patch('common.agent.job_logger.job_meta_context') as mock_context:
        mock_meta = {}
        mock_context.return_value.__enter__.return_value = mock_meta

        logger_agent._file_write(test_row)

        assert 'job_logger' in mock_meta
        assert mock_meta['job_logger'] == [test_row]


def test_log_step(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers

    logger_agent.log_step(
        gpu_rank=0,
        step_num=1,
        step_len=100,
        step_loss=0.5,
        step_lr=0.001,
        step_peak_memory_active=1000,
        step_peak_memory_alloc=2000,
        step_peak_memory_reserved=3000,
        step_time_elapsed_s=10.5,
        epoch_num=1,
        epoch_len=5
    )

    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'step'
    step_data = logged_data['data']
    assert step_data['gpu_rank'] == 0
    assert step_data['step_num'] == 1
    assert step_data['step_loss'] == 0.5
    assert step_data['step_lr'] == 0.001
    assert step_data['epoch_num'] == 1
    assert step_data['epoch_len'] == 5


def test_log_epoch(logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers

    logger_agent.log_epoch(
        gpu_rank=0,
        epoch_num=1,
        epoch_len=5,
        epoch_time_elapsed_s=100.5
    )

    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'epoch'
    epoch_data = logged_data['data']
    assert epoch_data['gpu_rank'] == 0
    assert epoch_data['epoch_num'] == 1
    assert epoch_data['epoch_len'] == 5
    assert epoch_data['epoch_time_elapsed_s'] == 100.5


@patch('common.agent.job_logger.get_results_bucket')
def test_log_weights(mock_get_bucket, logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers
    mock_get_bucket.return_value = 'test-bucket'

    weight_files = ['model1.pt', 'model2.pt']
    other_files = ['config.json']

    logger_agent.log_weights(weight_files, other_files)

    agent_logger.info.assert_called_once()
    logged_data = agent_logger.info.call_args[0][0]
    assert logged_data['operation'] == 'weights'
    weights_data = logged_data['data']
    assert weights_data['base_url'] == 'https://storage.googleapis.com/test-bucket/test-user/test-job'
    assert weights_data['weight_files'] == weight_files
    assert weights_data['other_files'] == other_files


def test_workflow_name(logger_agent):
    assert logger_agent._workflow_name == 'torchtunewrapper'


@patch('common.agent.job_logger.publish_to_pubsub')
@patch('common.agent.job_logger.insert_to_biqquery')
def test_multi_channel_logging(mock_insert_bq, mock_publish, logger_agent, mock_loggers):
    agent_logger, _ = mock_loggers
    test_data = {'key': 'value'}

    logger_agent._log_data('test_operation', test_data)

    # Verify all channels were logged to
    agent_logger.info.assert_called_once()
    mock_insert_bq.assert_called_once()
    mock_publish.assert_called_once()

