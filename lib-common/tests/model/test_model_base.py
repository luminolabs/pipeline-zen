import os
from unittest.mock import MagicMock, patch

import pytest

from common.model.base import BaseModelProvider, model_provider_factory
from common.model.hugging_face import HuggingFaceProvider


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def base_provider(mock_logger):
    return BaseModelProvider("hf://meta-llama/model", mock_logger)


@pytest.fixture
def mock_config(tmp_path):
    """Mock config with a temporary directory path"""
    with patch('common.model.base.config') as mock_cfg:
        mock_cfg.root_path = str(tmp_path)  # Use pytest's tmp_path
        mock_cfg.cache_dir = ".cache"
        yield mock_cfg


def test_base_provider_init(base_provider):
    """Test BaseModelProvider initialization"""
    assert base_provider.url == "hf://meta-llama/model"
    assert base_provider.provider == "hf"
    assert base_provider.model_name == "meta-llama/model"


def test_base_provider_get_model_cache_dir(base_provider, mock_config, tmp_path):
    """Test get_model_cache_dir method"""
    cache_dir = base_provider.get_model_cache_dir()
    expected_path = os.path.join(str(tmp_path), ".cache", "models", "hf", "meta-llama/model")
    assert cache_dir == expected_path
    assert os.path.exists(cache_dir)  # Verify directory was created


def test_base_provider_call_calls_fetch(base_provider):
    """Test that calling the provider calls fetch"""
    base_provider.fetch = MagicMock(return_value="test_path")
    result = base_provider()
    base_provider.fetch.assert_called_once()
    assert result == "test_path"


def test_base_provider_fetch_abstract(mock_logger):
    """Test that fetch raises NotImplementedError"""
    provider = BaseModelProvider("hf://test/model", mock_logger)
    with pytest.raises(NotImplementedError):
        provider.fetch()


def test_model_cache_dir_creation(base_provider, mock_config, tmp_path):
    """Test model cache directory creation with proper permissions"""
    # Get model cache directory
    cache_dir = base_provider.get_model_cache_dir()

    # Verify the directory was created with correct structure
    assert os.path.exists(cache_dir)
    assert cache_dir.startswith(str(tmp_path))
    assert "models/hf/meta-llama/model" in cache_dir


def test_model_provider_factory_huggingface(mock_logger):
    """Test factory returns HuggingFaceProvider for hf:// URLs"""
    provider = model_provider_factory(
        url="hf://meta-llama/model",
        logger=mock_logger
    )
    assert isinstance(provider, HuggingFaceProvider)


def test_model_provider_factory_invalid_url(mock_logger):
    """Test factory raises ValueError for invalid URLs"""
    with pytest.raises(ValueError, match=r"Unknown model provider: s3://bucket/model"):
        model_provider_factory(
            url="s3://bucket/model",
            logger=mock_logger
        )


def test_base_provider_with_kwargs(base_provider):
    """Test provider call with kwargs"""
    base_provider.fetch = MagicMock()
    base_provider(test_kwarg="value")
    base_provider.fetch.assert_called_once_with(test_kwarg="value")


def test_provider_url_parsing(mock_logger):
    """Test provider URL parsing with different formats"""
    test_cases = [
        ("hf://org/model", "hf", "org/model"),
        ("hf://org/model/v1", "hf", "org/model/v1"),
        ("hf://org/model-name-123", "hf", "org/model-name-123")
    ]

    for url, expected_provider, expected_model_name in test_cases:
        provider = BaseModelProvider(url, mock_logger)
        assert provider.provider == expected_provider
        assert provider.model_name == expected_model_name


def test_provider_maintains_state(mock_logger):
    """Test that provider maintains state between calls"""
    provider = BaseModelProvider(
        url="hf://org/model",
        logger=mock_logger
    )

    # Mock the fetch method to return different values
    provider.fetch = MagicMock()
    provider.fetch.side_effect = ["result1", "result2"]

    result1 = provider()
    result2 = provider()

    assert provider.fetch.call_count == 2
    assert result1 == "result1"
    assert result2 == "result2"


def test_provider_directory_creation_error(mock_logger, tmp_path):
    """Test handling of directory creation errors"""
    with patch('os.makedirs') as mock_makedirs:
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with patch('common.model.base.config') as mock_cfg:
            mock_cfg.root_path = str(tmp_path)
            mock_cfg.cache_dir = ".cache"

            with pytest.raises(PermissionError):
                BaseModelProvider("hf://org/model", mock_logger)


def test_model_provider_factory_none_url(mock_logger):
    """Test factory handles None URL"""
    with pytest.raises(AttributeError):
        model_provider_factory(None, mock_logger)


def test_model_provider_factory_empty_url(mock_logger):
    """Test factory handles empty URL"""
    with pytest.raises(ValueError, match="Unknown model provider: "):
        model_provider_factory("", mock_logger)


def test_cache_dir_nested_creation(base_provider, mock_config, tmp_path):
    """Test creation of deeply nested cache directories"""
    # Mock a deeply nested model name
    provider = BaseModelProvider("hf://org/nested/model/path", mock_logger)

    cache_dir = provider.get_model_cache_dir()
    assert os.path.exists(cache_dir)
    assert "models/hf/org/nested/model/path" in cache_dir
