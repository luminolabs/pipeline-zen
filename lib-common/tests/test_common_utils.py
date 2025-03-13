import json
import logging
import os
from datetime import datetime, timezone
from unittest.mock import patch, mock_open

import pytest
from omegaconf import DictConfig, OmegaConf

from common.utils import (
    utcnow, utcnow_str, JsonFormatter, setup_logger, get_work_dir,
    job_meta_context, JsonEnumBase, AutoJSONEncoder, save_job_results,
    is_local_env, load_job_config, read_job_config_from_file, get_artifacts,
    SYSTEM_TIMESTAMP_FORMAT
)

# Test Data
TEST_JOB_ID = "test-job-123"
TEST_USER_ID = "test-user-456"


# Fixtures
@pytest.fixture
def mock_config():
    with patch('common.utils.config') as mock_cfg:
        mock_cfg.env_name = 'test'
        mock_cfg.local_env_name = 'local'
        mock_cfg.root_path = '/test/root'
        mock_cfg.work_dir = '.results'
        mock_cfg.job_configs_path = 'job-configs'
        mock_cfg.log_stdout = True
        mock_cfg.log_level = logging.INFO
        mock_cfg.started_file = '.started'
        mock_cfg.finished_file = '.finished'
        mock_cfg.job_meta_file = 'job-meta.json'
        yield mock_cfg


# Time Utilities Tests
def test_utcnow():
    """Test utcnow returns current UTC time without timezone info"""
    result = utcnow()
    assert isinstance(result, datetime)
    assert result.tzinfo is None
    # Test the time is close to current time
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((result - now).total_seconds()) < 1


def test_utcnow_str():
    """Test utcnow_str returns formatted UTC time string"""
    result = utcnow_str()
    # Verify the format matches SYSTEM_TIMESTAMP_FORMAT
    try:
        datetime.strptime(result, SYSTEM_TIMESTAMP_FORMAT)
    except ValueError:
        pytest.fail("utcnow_str returned invalid datetime format")


def test_utcnow_str_custom_format():
    """Test utcnow_str with custom format"""
    custom_format = "%Y-%m-%d"
    result = utcnow_str(fmt=custom_format)
    try:
        datetime.strptime(result, custom_format)
    except ValueError:
        pytest.fail("utcnow_str with custom format returned invalid datetime format")


# JsonEnumBase Tests
class SampleEnum(JsonEnumBase):
    """Sample enum for testing JsonEnumBase serialization"""
    VALUE1 = "test1"
    VALUE2 = "test2"


def test_json_enum_base():
    """Test JsonEnumBase serialization"""
    assert SampleEnum.VALUE1.json() == "test1"
    assert SampleEnum.VALUE2.json() == "test2"


# AutoJSONEncoder Tests
def test_auto_json_encoder_enum():
    """Test AutoJSONEncoder with Enum values"""
    data = {"enum": SampleEnum.VALUE1}
    result = json.dumps(data, cls=AutoJSONEncoder)
    assert '"enum": "test1"' in result


def test_auto_json_encoder_dict_config():
    """Test AutoJSONEncoder with DictConfig"""
    config = OmegaConf.create({"key": "value"})
    result = json.dumps({"config": config}, cls=AutoJSONEncoder)
    assert '"config": "{\\"key\\": \\"value\\"}"' in result


# JsonFormatter Tests
def test_json_formatter():
    """Test JsonFormatter formats log records correctly"""
    formatter = JsonFormatter(TEST_JOB_ID, TEST_USER_ID)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None
    )

    result = json.loads(formatter.format(record))
    assert result["job_id"] == TEST_JOB_ID
    assert result["user_id"] == TEST_USER_ID
    assert result["message"] == "test message"
    assert result["level"] == "INFO"


# File Utilities Tests
@patch('os.makedirs')
def test_get_work_dir(mock_makedirs, mock_config):
    """Test work directory path construction"""
    expected_path = os.path.join(mock_config.root_path, mock_config.work_dir, TEST_USER_ID, TEST_JOB_ID)

    result = get_work_dir(TEST_JOB_ID, TEST_USER_ID)

    assert result == expected_path
    mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


# Job Utilities Tests
@patch('builtins.open', new_callable=mock_open)
def test_save_job_results(mock_file, mock_config):
    """Test saving job results to file"""
    results = {"status": "success", "metrics": [1, 2, 3]}
    with patch('common.utils.get_work_dir') as mock_get_work_dir:
        mock_get_work_dir.return_value = "/test/path"
        save_job_results(TEST_JOB_ID, TEST_USER_ID, results, "test_job")

        mock_file.assert_called_once_with("/test/path/test_job.json", "w")
        written_data = json.loads(mock_file().write.call_args[0][0])
        assert written_data == results


# Configuration Utilities Tests
def test_is_local_env(mock_config):
    """Test environment type checking"""
    mock_config.env_name = 'local'
    assert is_local_env() is True

    mock_config.env_name = 'prod'
    assert is_local_env() is False


def test_load_job_config(mock_config):
    """Test loading job configuration"""
    with patch('common.utils.read_job_config_from_file') as mock_read:
        mock_read.return_value = DictConfig({"test": "config"})
        result = load_job_config("test_config")
        assert isinstance(result, DictConfig)
        assert result["test"] == "config"


def test_read_job_config_from_file_not_found(mock_config):
    """Test reading non-existent job config file"""
    with pytest.raises(FileNotFoundError):
        read_job_config_from_file("nonexistent_config")


# Artifact Tests
@patch('os.listdir')
@patch('common.utils.get_work_dir')
def test_get_artifacts(mock_get_work_dir, mock_listdir):
    """Test getting job artifacts"""
    mock_get_work_dir.return_value = "/test/path"
    mock_listdir.side_effect = [
        ["epoch_0"],  # First call
        ["model.safetensors"],  # Second call
        ["config.json"]  # Third call
    ]

    weight_files, other_files = get_artifacts(TEST_JOB_ID, TEST_USER_ID)

    assert weight_files == ["epoch_0/model.safetensors"]
    assert other_files == ["epoch_0/config.json"]


# Error Cases
def test_setup_logger_invalid_level(mock_config, tmp_path):
    """Test logger setup with invalid log level"""
    with patch('common.utils.get_work_dir') as mock_get_work_dir:
        mock_get_work_dir.return_value = str(tmp_path)
        mock_config.log_level = None
        logger = setup_logger("test_logger", TEST_JOB_ID, TEST_USER_ID)
        assert logger.level == logging.INFO  # Should default to INFO


def test_json_formatter_invalid_message():
    """Test JsonFormatter with invalid JSON string message"""
    formatter = JsonFormatter(TEST_JOB_ID, TEST_USER_ID)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="{invalid json",
        args=(),
        exc_info=None
    )

    result = json.loads(formatter.format(record))
    assert result["message"] == "{invalid json"  # Should return original string


@patch('builtins.open')
def test_job_meta_context_file_error(mock_file, mock_config):
    """Test job metadata context manager with file error"""
    mock_file.side_effect = IOError("Test error")

    with pytest.raises(IOError):
        with job_meta_context(TEST_JOB_ID, TEST_USER_ID):
            pass
