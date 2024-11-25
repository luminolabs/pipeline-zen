import os
from unittest.mock import MagicMock, patch

import pytest

from common.dataset.base import BaseDatasetProvider, dataset_provider_factory
from common.dataset.file_system import FileSystemProvider
from common.dataset.gcp_bucket import GcpBucketProvider
from common.utils import get_work_dir


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def base_provider(mock_logger):
    return BaseDatasetProvider(
        url="test://example.com/dataset",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )


def test_base_provider_init(base_provider):
    """Test BaseDatasetProvider initialization"""
    assert base_provider.url == "test://example.com/dataset"
    assert base_provider.job_id == "test-job"
    assert base_provider.user_id == "test-user"
    assert base_provider.local_path == os.path.join(get_work_dir("test-job", "test-user"), "dataset.jsonl")


def test_base_provider_call_calls_fetch(base_provider):
    """Test that calling the provider calls fetch"""
    base_provider.fetch = MagicMock(return_value="test_path")
    result = base_provider()
    base_provider.fetch.assert_called_once()
    assert result == "test_path"


def test_base_provider_fetch_abstract(mock_logger):
    """Test that fetch raises NotImplementedError"""
    provider = BaseDatasetProvider(
        url="test://example.com/dataset",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )
    with pytest.raises(NotImplementedError):
        provider.fetch()


@patch('common.dataset.base.get_work_dir')
def test_local_path_construction(mock_get_work_dir, mock_logger):
    """Test local path construction"""
    mock_get_work_dir.return_value = "/test/path"
    provider = BaseDatasetProvider("test://url", "job1", "user1", mock_logger)
    assert provider.local_path == "/test/path/dataset.jsonl"


def test_dataset_provider_factory_gcp(mock_logger):
    """Test factory returns GcpBucketProvider for gs:// URLs"""
    provider = dataset_provider_factory(
        url="gs://bucket/path/dataset",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )
    assert isinstance(provider, GcpBucketProvider)


def test_dataset_provider_factory_file(mock_logger):
    """Test factory returns FileSystemProvider for file:// URLs"""
    provider = dataset_provider_factory(
        url="file://path/to/dataset",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )
    assert isinstance(provider, FileSystemProvider)


def test_dataset_provider_factory_invalid_url(mock_logger):
    """Test factory raises ValueError for invalid URLs"""
    with pytest.raises(ValueError, match=r"Unknown dataset provider: http://example.com/dataset"):
        dataset_provider_factory(
            url="http://example.com/dataset",
            job_id="test-job",
            user_id="test-user",
            logger=mock_logger
        )


def test_base_provider_with_kwargs(base_provider):
    """Test provider call with kwargs"""
    base_provider.fetch = MagicMock()
    base_provider(test_kwarg="value")
    base_provider.fetch.assert_called_once_with(test_kwarg="value")
