import os
from unittest.mock import patch, MagicMock

import pytest
from omegaconf import DictConfig
from torch.distributed.launcher import LaunchConfig

from torchtunewrapper.workflow import _download_dataset, _download_model, run, main


@pytest.fixture
def mock_config(monkeypatch):
    """Setup mock configuration singleton"""
    # Mock config properties
    monkeypatch.setattr('common.config_manager.config.mock_user_has_enough_credits', False)
    monkeypatch.setattr('common.config_manager.config.customer_api_enabled', True)
    monkeypatch.setattr('common.config_manager.config.customer_api_url', "http://test-api.com")
    monkeypatch.setattr('common.config_manager.config.customer_api_credits_deduct_endpoint', "/credits/deduct")
    monkeypatch.setattr('common.config_manager.config.customer_api_key', "test-key")
    monkeypatch.setattr('common.config_manager.config.local_env_name', "local")
    monkeypatch.setattr('common.config_manager.config.env', "test")
    monkeypatch.setattr('common.config_manager.config.job_meta_file', "job-meta.json")
    monkeypatch.setattr('common.config_manager.config.root_path', ".")
    monkeypatch.setattr('common.config_manager.config.work_dir', ".results")
    monkeypatch.setattr('common.config_manager.config.job_configs_path', "job-configs")


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_job_config():
    return DictConfig({
        'job_id': 'test-job',
        'user_id': 'test-user',
        'model_base': 'hf://meta-llama/Llama-3.1-8B-Instruct',  # Use supported model base
        'dataset_id': 'test-dataset',
        'num_gpus': 1,
        'use_lora': True,
        'use_qlora': False,
        'name': 'test-name'
    })


@pytest.fixture
def mock_tt_config():
    return DictConfig({
        'base_model_path': '/test/model/path',
        'dataset_id': 'test-dataset',
        'shuffle': True,
        'batch_size': 4,
        'lr': 0.001,
        'seed': 42
    })


# Test _download_dataset
@patch('torchtunewrapper.workflow.chat_dataset')
@patch('torchtunewrapper.workflow.dataset_provider_factory')
def test_download_dataset(mock_provider_factory, mock_chat_dataset, mock_logger, mock_config):
    """Test dataset download functionality"""
    # Setup mocks
    mock_provider = MagicMock()
    mock_provider.return_value = "/test/dataset/path"
    mock_provider_factory.return_value = mock_provider

    mock_dataset = MagicMock()
    mock_chat_dataset.return_value = mock_dataset

    # Create test config with required fields
    test_config = DictConfig({
        'dataset_id': 'test-dataset',
        'job_id': 'test-job',
        'user_id': 'test-user',
        'split': 'train',
        'max_seq_len': 512,
        'train_on_input': False,
        'packed': False
    })

    with patch('common.heartbeats.send_heartbeat'):
        # Call function
        result = _download_dataset(test_config, mock_logger)

    # Verify
    assert result == mock_dataset
    mock_provider_factory.assert_called_once_with(
        url='test-dataset',
        job_id='test-job',
        user_id='test-user',
        logger=mock_logger
    )
    mock_chat_dataset.assert_called_once()


# Test _download_model
@patch('torchtunewrapper.workflow.model_provider_factory')
def test_download_model(mock_provider_factory, mock_logger, mock_config):
    """Test model download functionality"""
    # Setup mock
    mock_provider = MagicMock()
    mock_provider.return_value = "/test/model/path"
    mock_provider_factory.return_value = mock_provider

    with patch('common.heartbeats.send_heartbeat'):
        # Call function with job_id and user_id
        result = _download_model("hf://test-model", mock_logger)

    # Verify
    assert result == "/test/model/path"
    mock_provider_factory.assert_called_once()


# Test run function success cases
@patch('torchtunewrapper.workflow.threading.Thread')
@patch('torchtunewrapper.workflow._download_dataset')
@patch('torchtunewrapper.workflow._download_model')
@patch('torchtunewrapper.workflow.import_torchtune_recipe_fn')
def test_run_single_device(mock_import_fn, mock_download_model, mock_download_dataset,
                           mock_thread, mock_job_config, mock_tt_config, mock_logger, mock_config):
    """Test successful run with single device"""
    # Setup mocks
    mock_dataset = MagicMock()
    mock_download_dataset.return_value = mock_dataset
    mock_download_model.return_value = "/test/model/path"

    # Create mock recipe with proper name attribute
    mock_recipe = MagicMock()
    mock_recipe.__name__ = "test_recipe"
    mock_import_fn.return_value = mock_recipe

    # Call function
    result = run(
        mock_job_config.job_id,
        mock_job_config.user_id,
        mock_job_config,
        mock_tt_config,
        mock_logger
    )

    # Verify
    assert result is True
    mock_download_dataset.assert_called_once()
    mock_download_model.assert_called_once()
    mock_import_fn.assert_called_once()
    mock_recipe.assert_called_once()


@patch('torchtunewrapper.workflow.threading.Thread')
@patch('torchtunewrapper.workflow._download_dataset')
@patch('torchtunewrapper.workflow._download_model')
@patch('torchtunewrapper.workflow.import_torchtune_recipe_fn')
@patch('torchtunewrapper.workflow.elastic_launch')
def test_run_multi_device(mock_elastic_launch, mock_import_fn, mock_download_model,
                          mock_download_dataset, mock_thread, mock_job_config,
                          mock_tt_config, mock_logger, mock_config):
    """Test successful run with multiple devices"""
    # Setup mocks
    mock_dataset = MagicMock()
    mock_download_dataset.return_value = mock_dataset
    mock_download_model.return_value = "/test/model/path"

    # Mock recipe with proper name attribute
    mock_recipe = MagicMock()
    mock_recipe.__name__ = "test_recipe"
    mock_import_fn.return_value = mock_recipe

    mock_launch_fn = MagicMock()
    mock_elastic_launch.return_value = mock_launch_fn

    # Update config for multiple devices
    mock_job_config.num_gpus = 2

    # Call function
    result = run(
        mock_job_config.job_id,
        mock_job_config.user_id,
        mock_job_config,
        mock_tt_config,
        mock_logger
    )

    # Verify
    assert result is True
    mock_elastic_launch.assert_called_once()
    launch_config = mock_elastic_launch.call_args[1]['config']
    assert isinstance(launch_config, LaunchConfig)
    assert launch_config.min_nodes == 1
    assert launch_config.max_nodes == 1
    assert launch_config.nproc_per_node == 2
    mock_launch_fn.assert_called_once()


# Test run function error cases
@patch('torchtunewrapper.workflow._download_dataset')
def test_run_download_dataset_error(mock_download_dataset, mock_job_config,
                                    mock_tt_config, mock_logger, mock_config):
    """Test handling of dataset download error"""
    # Setup mock to raise error and capture logs
    mock_download_dataset.side_effect = Exception("Dataset download failed")

    with pytest.raises(Exception) as exc_info:
        run(mock_job_config.job_id, mock_job_config.user_id,
            mock_job_config, mock_tt_config, mock_logger)

    assert str(exc_info.value) == "Dataset download failed"


# Test main function
@patch('torchtunewrapper.workflow.load_job_config')
@patch('torchtunewrapper.workflow.setup_logger')
@patch('torchtunewrapper.workflow.read_job_config_from_file')
@patch('torchtunewrapper.workflow.OmegaConf.load')
@patch('torchtunewrapper.workflow.run')
def test_main_success(mock_run, mock_load, mock_read_config, mock_setup_logger,
                      mock_load_config, mock_logger, mock_config):
    """Test successful execution of main function"""
    # Setup mocks
    mock_load_config.return_value = DictConfig({
        'model_base': 'hf://meta-llama/Llama-3.1-8B-Instruct',  # Use supported model
        'num_gpus': 1
    })
    mock_read_config.return_value = DictConfig({'test': 'config'})
    mock_setup_logger.return_value = mock_logger
    mock_run.return_value = True
    mock_load.return_value = {}

    # Call function
    result = main(
        job_id="test-job",
        user_id="test-user",
        job_config_name="test_config",
        dataset_id="test-dataset",
        batch_size=4,
        shuffle=True,
        num_epochs=1,
        use_lora=True,
        use_qlora=False,
        lr=0.001,
        seed=42,
        num_gpus=1
    )

    # Verify
    assert result is True
    mock_load_config.assert_called_once()
    mock_setup_logger.assert_called_once()
    mock_read_config.assert_called_once()
    mock_run.assert_called_once()


@patch('torchtunewrapper.workflow.setup_logger')
@patch('torchtunewrapper.workflow.load_job_config')
def test_main_error(mock_load_config, mock_setup_logger, mock_logger):
    """Test error handling in main function"""
    # Setup mocks
    mock_setup_logger.return_value = mock_logger
    mock_load_config.side_effect = Exception("Config loading failed")

    # Run test
    with pytest.raises(Exception) as exc_info:
        main(
            job_id="test-job",
            user_id="test-user",
            job_config_name="test_config"
        )

    assert str(exc_info.value) == "Config loading failed"


@patch('torchtunewrapper.workflow.load_job_config')
@patch('torchtunewrapper.workflow.OmegaConf.load')
def test_main_environment_handling(mock_load, mock_load_config, mock_config):
    """Test environment variable handling in workflow"""
    env_vars = {
        'PZ_HUGGINGFACE_TOKEN': 'test-token',
        'PZ_ENV': 'test',
        'PZ_CUSTOMER_API_KEY': 'test-key'
    }

    # Setup mocks
    mock_load_config.return_value = DictConfig({
        'model_base': 'hf://meta-llama/Llama-3.1-8B-Instruct',
        'num_gpus': 1
    })
    mock_load.return_value = {}

    with patch.dict(os.environ, env_vars, clear=True):
        with patch('torchtunewrapper.workflow.run') as mock_run:
            mock_run.return_value = True

            result = main(
                job_id="test-job",
                user_id="test-user",
                job_config_name="test_config"
            )

            assert result is True
            assert os.environ.get('PZ_HUGGINGFACE_TOKEN') == 'test-token'
            assert os.environ.get('PZ_ENV') == 'test'
            assert os.environ.get('PZ_CUSTOMER_API_KEY') == 'test-key'
