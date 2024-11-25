from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from common.heartbeats import heartbeat_wrapper, send_heartbeat


@pytest.fixture
def mock_config():
    with patch('common.heartbeats.config') as mock_cfg:
        mock_cfg.heartbeat_topic = 'test-topic'
        mock_cfg.job_id = 'test-job-id'
        mock_cfg.user_id = 'test-user-id'
        yield mock_cfg


@pytest.fixture
def mock_job_meta_context():
    with patch('common.heartbeats.job_meta_context') as mock_ctx:
        # Create a mock context manager
        mock_context = MagicMock()
        mock_dict = {}
        mock_context.__enter__.return_value = mock_dict
        mock_ctx.return_value = mock_context
        yield mock_ctx, mock_dict


@pytest.fixture
def mock_get_vm_name():
    with patch('common.heartbeats.get_vm_name_from_metadata') as mock_vm:
        mock_vm.return_value = 'test-vm'
        yield mock_vm


@pytest.fixture
def mock_pubsub():
    with patch('common.heartbeats.publish_to_pubsub') as mock_pub:
        yield mock_pub


def test_send_heartbeat_basic(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test basic heartbeat sending with minimal parameters"""
    mock_ctx, mock_dict = mock_job_meta_context

    send_heartbeat('test-job', 'test-user', 'test-status')

    # Check job meta was updated correctly
    assert 'heartbeats' in mock_dict
    assert len(mock_dict['heartbeats']) == 1
    heartbeat = mock_dict['heartbeats'][0]
    assert heartbeat['status'] == 'test-status'
    assert heartbeat['vm_name'] == 'test-vm'
    assert heartbeat['elapsed_time_s'] is None
    assert 'timestamp' in heartbeat

    # Verify PubSub was called correctly
    mock_pubsub.assert_called_once_with(
        'test-job', 'test-user',
        'test-topic',
        heartbeat
    )


def test_send_heartbeat_with_elapsed_time(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test heartbeat sending with elapsed time"""
    mock_ctx, mock_dict = mock_job_meta_context

    send_heartbeat('test-job', 'test-user', 'test-status', elapsed_time=123.456)

    heartbeat = mock_dict['heartbeats'][0]
    assert heartbeat['elapsed_time_s'] == '123.46'  # Verify formatting


def test_heartbeat_wrapper_successful_execution(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test heartbeat wrapper with successful function execution"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func():
        return True

    result = test_func()

    assert result is True
    assert mock_pubsub.call_count == 3  # start, finish, total

    # Verify the sequence of heartbeats
    calls = mock_pubsub.call_args_list
    assert calls[0][0][2] == 'test-topic'  # topic name
    assert calls[0][0][3]['status'] == 'wf-test-workflow-test-task-start'
    assert calls[1][0][3]['status'] == 'wf-test-workflow-test-task-finish'
    assert calls[2][0][3]['status'] == 'wf-test-workflow-test-task-total'

    # Verify elapsed time was included in total
    assert 'elapsed_time_s' in calls[2][0][3]


def test_heartbeat_wrapper_function_returns_false(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test heartbeat wrapper when function returns False"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func():
        return False

    result = test_func()

    assert result is False
    assert mock_pubsub.call_count == 3  # start, error, total

    calls = mock_pubsub.call_args_list
    assert calls[1][0][3]['status'] == 'wf-test-workflow-test-task-error'


def test_heartbeat_wrapper_function_raises_exception(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test heartbeat wrapper when function raises an exception"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        test_func()

    assert mock_pubsub.call_count == 2  # start, error, no finish

    calls = mock_pubsub.call_args_list
    assert calls[1][0][3]['status'] == 'wf-test-workflow-test-task-error'


def test_heartbeat_wrapper_preserves_function_args(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test heartbeat wrapper preserves function arguments"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func(arg1, arg2, kwarg1='default'):
        return f"{arg1}-{arg2}-{kwarg1}"

    result = test_func('val1', 'val2', kwarg1='custom')

    assert result == 'val1-val2-custom'


def test_heartbeat_wrapper_timestamp_format(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test timestamp format in heartbeat messages"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func():
        return True

    with patch('common.heartbeats.utcnow_str') as mock_time:
        mock_time.return_value = '2024-01-01T12:00:00Z'
        test_func()

        calls = mock_pubsub.call_args_list
        for call in calls:
            assert call[0][3]['timestamp'] == '2024-01-01T12:00:00Z'


def test_heartbeat_wrapper_elapsed_time_calculation(mock_config, mock_job_meta_context, mock_get_vm_name, mock_pubsub):
    """Test elapsed time calculation in heartbeat wrapper"""

    @heartbeat_wrapper('test-workflow', 'test-task')
    def test_func():
        # Simulate some work
        return True

    # Mock time to control elapsed time calculation
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(seconds=5)

    with patch('common.heartbeats.utcnow') as mock_time:
        mock_time.side_effect = [start_time, end_time, end_time]  # For start and end time calculations
        test_func()

        # Check the elapsed time in the total heartbeat
        total_heartbeat = mock_pubsub.call_args_list[-1][0][3]
        assert float(total_heartbeat['elapsed_time_s']) == 5.0
