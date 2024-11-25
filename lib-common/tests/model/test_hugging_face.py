import os
from unittest.mock import patch, MagicMock

import pytest

from common.model.hugging_face import HuggingFaceProvider


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_config():
    with patch('common.model.hugging_face.config') as mock_cfg:
        mock_cfg.huggingface_token = 'test-token'
        yield mock_cfg


@pytest.fixture
def provider(mock_logger, mock_config):
    return HuggingFaceProvider('hf://meta-llama/model', mock_logger)


@pytest.fixture
def mock_snapshot_download():
    with patch('common.model.hugging_face.huggingface_hub.snapshot_download') as mock_download:
        mock_download.return_value = '/test/download/path'
        yield mock_download


def test_huggingface_provider_init(provider):
    """Test HuggingFaceProvider initialization"""
    assert provider.url == 'hf://meta-llama/model'
    assert provider.provider == 'hf'
    assert provider.model_name == 'meta-llama/model'


def test_fetch_downloads_model(provider, mock_logger, mock_snapshot_download, mock_config):
    """Test that fetch downloads the model with correct parameters"""
    # When
    result = provider.fetch()

    # Then
    mock_snapshot_download.assert_called_once_with(
        'meta-llama/model',
        token='test-token',
        cache_dir=provider.local_path
    )
    assert result == '/test/download/path'
    mock_logger.info.assert_called_once_with('Downloading Hugging Face model: hf://meta-llama/model')


def test_fetch_with_additional_kwargs(provider, mock_snapshot_download):
    """Test that fetch passes additional kwargs to snapshot_download"""
    # When
    provider.fetch(local_files_only=True, resume_download=False)

    # Then
    mock_snapshot_download.assert_called_once_with(
        'meta-llama/model',
        token='test-token',
        cache_dir=provider.local_path,
        local_files_only=True,
        resume_download=False
    )


def test_fetch_without_token(provider, mock_snapshot_download, mock_config):
    """Test fetch behavior when no token is provided"""
    # Given
    mock_config.huggingface_token = None

    # When
    result = provider.fetch()

    # Then
    mock_snapshot_download.assert_called_once_with(
        'meta-llama/model',
        token=None,
        cache_dir=provider.local_path
    )
    assert result == '/test/download/path'


@patch('common.model.hugging_face.huggingface_hub.snapshot_download')
def test_fetch_handles_download_error(mock_download, provider, mock_logger):
    """Test that fetch handles download errors appropriately"""
    # Given
    mock_download.side_effect = Exception("Download failed")

    # When/Then
    with pytest.raises(Exception, match="Download failed"):
        provider.fetch()


def test_model_cache_path_creation(provider, tmp_path):
    """Test that the model cache directory is created correctly"""
    # Given
    with patch('common.model.base.config') as mock_base_config:
        mock_base_config.root_path = str(tmp_path)
        mock_base_config.cache_dir = ".cache"

        # When
        cache_dir = provider.get_model_cache_dir()

        # Then
        assert cache_dir == os.path.join(str(tmp_path), ".cache", "models", "hf", "meta-llama/model")
        assert os.path.exists(cache_dir)


def test_provider_call_invokes_fetch(provider):
    """Test that calling the provider invokes fetch"""
    # Given
    provider.fetch = MagicMock(return_value="/test/path")

    # When
    result = provider()

    # Then
    provider.fetch.assert_called_once()
    assert result == "/test/path"


def test_provider_maintains_state(mock_logger, mock_config):
    """Test that provider maintains state between calls"""
    # Given
    provider = HuggingFaceProvider('hf://org/model', mock_logger)

    with patch('common.model.hugging_face.huggingface_hub.snapshot_download') as mock_download:
        mock_download.side_effect = ["/path1", "/path2"]

        # When
        result1 = provider()
        result2 = provider()

        # Then
        assert mock_download.call_count == 2
        assert result1 == "/path1"
        assert result2 == "/path2"


def test_provider_with_different_model_names(mock_logger, mock_config, mock_snapshot_download):
    """Test provider handles different model name formats"""
    test_cases = [
        "hf://org/model",
        "hf://org/model-v1",
        "hf://org/model-name-123",
        "hf://org/model/version1"
    ]

    for url in test_cases:
        provider = HuggingFaceProvider(url, mock_logger)
        provider.fetch()

        expected_model_name = url[5:]  # Remove 'hf://' prefix
        mock_snapshot_download.assert_called_with(
            expected_model_name,
            token='test-token',
            cache_dir=provider.local_path
        )
