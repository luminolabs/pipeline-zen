import os
from unittest.mock import MagicMock, patch

import pytest

from common.dataset.gcp_bucket import GcpBucketProvider
from common.utils import get_work_dir


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def valid_bucket_provider(mock_logger):
    return GcpBucketProvider(
        url="gs://lum-local-pipeline-zen-datasets/user123/dataset.jsonl",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )


@pytest.fixture
def invalid_bucket_provider(mock_logger):
    return GcpBucketProvider(
        url="gs://invalid-bucket/datasets/user123/dataset.jsonl",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )


def test_bucket_provider_init(valid_bucket_provider):
    """Test GcpBucketProvider initialization"""
    assert valid_bucket_provider.url == "gs://lum-local-pipeline-zen-datasets/user123/dataset.jsonl"
    assert valid_bucket_provider.job_id == "test-job"
    assert valid_bucket_provider.user_id == "test-user"
    assert valid_bucket_provider.local_path == os.path.join(get_work_dir("test-job", "test-user"), "dataset.jsonl")


def test_bucket_provider_fetch_valid_bucket(valid_bucket_provider, mock_logger):
    """Test successful dataset fetch from valid bucket"""
    with patch('common.dataset.gcp_bucket.download_object') as mock_download:
        result = valid_bucket_provider.fetch()

        # Verify download was called with correct parameters
        mock_download.assert_called_once_with(
            'lum-local-pipeline-zen-datasets',
            'user123/dataset.jsonl',
            valid_bucket_provider.local_path
        )

        # Verify log message
        mock_logger.info.assert_called_once_with(
            'Downloading GCP bucket dataset: gs://lum-local-pipeline-zen-datasets/user123/dataset.jsonl'
        )

        # Verify correct local path is returned
        assert result == valid_bucket_provider.local_path


def test_bucket_provider_fetch_invalid_bucket(invalid_bucket_provider):
    """Test fetch from invalid bucket raises ValueError"""
    with pytest.raises(ValueError, match=r'Upload datasets to `gs://lum-local-pipeline-zen-datasets/<user_id>` only'):
        invalid_bucket_provider.fetch()


def test_bucket_provider_call_invokes_fetch(valid_bucket_provider):
    """Test that calling the provider invokes fetch"""
    with patch.object(valid_bucket_provider, 'fetch') as mock_fetch:
        mock_fetch.return_value = "/test/path/dataset.jsonl"
        result = valid_bucket_provider()

        mock_fetch.assert_called_once()
        assert result == "/test/path/dataset.jsonl"


@patch('common.dataset.gcp_bucket.download_object')
def test_bucket_provider_download_error(mock_download, valid_bucket_provider):
    """Test handling of download errors"""
    mock_download.side_effect = Exception("Download failed")

    with pytest.raises(Exception, match="Download failed"):
        valid_bucket_provider.fetch()


def test_bucket_provider_with_different_paths(mock_logger):
    """Test provider handles different bucket paths correctly"""
    test_cases = [
        "gs://lum-local-pipeline-zen-datasets/user123/data.jsonl",
        "gs://lum-local-pipeline-zen-datasets/user123/subfolder/data.jsonl",
        "gs://lum-local-pipeline-zen-datasets/user123/deep/nested/data.jsonl"
    ]

    for test_path in test_cases:
        provider = GcpBucketProvider(
            url=test_path,
            job_id="test-job",
            user_id="test-user",
            logger=mock_logger
        )
        with patch('common.dataset.gcp_bucket.download_object') as mock_download:
            provider.fetch()
            mock_download.assert_called_once_with(
                'lum-local-pipeline-zen-datasets',
                '/'.join(test_path.split('/')[3:]),
                provider.local_path
            )


def test_bucket_provider_with_kwargs(valid_bucket_provider):
    """Test provider handles additional kwargs correctly"""
    test_kwargs = {'test_param': True, 'another_param': 'value'}

    with patch.object(valid_bucket_provider, 'fetch') as mock_fetch:
        valid_bucket_provider(**test_kwargs)
        mock_fetch.assert_called_once_with(**test_kwargs)


def test_bucket_provider_maintains_state(mock_logger):
    """Test that provider maintains state between calls"""
    provider = GcpBucketProvider(
        url="gs://lum-pipeline-zen-jobs-us/datasets/user123/dataset.jsonl",
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
def test_bucket_provider_local_path_creation(mock_get_work_dir, mock_logger):
    """Test local path creation for downloaded dataset"""
    mock_get_work_dir.return_value = "/test/work/dir"
    provider = GcpBucketProvider(
        url="gs://lum-pipeline-zen-jobs-us/datasets/user123/dataset.jsonl",
        job_id="test-job",
        user_id="test-user",
        logger=mock_logger
    )

    assert provider.local_path == os.path.join("/test/work/dir", "dataset.jsonl")
