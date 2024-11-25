import os
from unittest.mock import MagicMock, patch

import pytest

from common.dataset.file_system import FileSystemProvider
from common.utils import get_work_dir


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def file_provider(mock_logger):
    return FileSystemProvider(
        url="file:///path/to/dataset.jsonl",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )


def test_file_provider_init(file_provider):
    """Test FileSystemProvider initialization"""
    assert file_provider.url == "file:///path/to/dataset.jsonl"
    assert file_provider.job_id == "test-job"
    assert file_provider.user_id == "test-user"
    assert file_provider.local_path == os.path.join(get_work_dir("test-job", "test-user"), "dataset.jsonl")


def test_file_provider_fetch_returns_local_path(file_provider):
    """Test that fetch returns the correct local path"""
    expected_path = os.path.join("/test/work/dir", "dataset.jsonl")
    file_provider.local_path = expected_path

    result = file_provider.fetch()

    assert result == expected_path


def test_file_provider_call_invokes_fetch(file_provider):
    """Test that calling the provider invokes fetch"""
    with patch.object(file_provider, 'fetch') as mock_fetch:
        mock_fetch.return_value = "/test/path/dataset.jsonl"
        result = file_provider()

        mock_fetch.assert_called_once()
        assert result == "/test/path/dataset.jsonl"


@patch('common.dataset.base.get_work_dir')
def test_file_provider_with_different_paths(mock_get_work_dir, mock_logger):
    """Test provider handles different file paths correctly"""
    test_cases = [
        "file:///absolute/path/data.jsonl",
        "file://./relative/path/data.jsonl",
        "file://~/home/path/data.jsonl",
        "file://data.jsonl"
    ]

    mock_get_work_dir.return_value = "/test/work/dir"

    for test_path in test_cases:
        provider = FileSystemProvider(
            url=test_path,
            job_id="test-job",
            user_id="test-user",
            logger=mock_logger
        )
        assert provider.fetch() == os.path.join("/test/work/dir", "dataset.jsonl")


def test_file_provider_with_kwargs(file_provider):
    """Test provider handles additional kwargs correctly"""
    with patch.object(file_provider, 'fetch') as mock_fetch:
        file_provider(test_kwarg="value")
        mock_fetch.assert_called_once_with(test_kwarg="value")


def test_file_provider_maintains_state(mock_logger):
    """Test that provider maintains state between calls"""
    provider = FileSystemProvider(
        url="file:///path/to/dataset.jsonl",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )

    with patch.object(provider, 'fetch') as mock_fetch:
        mock_fetch.return_value = "/test/path1"
        result1 = provider()

        mock_fetch.return_value = "/test/path2"
        result2 = provider()

        assert result1 == "/test/path1"
        assert result2 == "/test/path2"
        assert mock_fetch.call_count == 2


@patch('common.dataset.base.get_work_dir')
def test_file_provider_with_empty_path(mock_get_work_dir, mock_logger):
    """Test provider handles empty file paths appropriately"""
    mock_get_work_dir.return_value = "/test/work/dir"

    provider = FileSystemProvider(
        url="file://",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )

    assert provider.fetch() == os.path.join("/test/work/dir", "dataset.jsonl")


def test_file_provider_fetch_with_custom_kwargs(file_provider):
    """Test fetch method handles custom kwargs"""
    result = file_provider.fetch(custom_param=True, another_param="test")
    assert result == file_provider.local_path
