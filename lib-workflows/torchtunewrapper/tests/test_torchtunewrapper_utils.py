from unittest.mock import patch, MagicMock

import pytest
from omegaconf import DictConfig

from torchtunewrapper.utils import (
    import_torchtune_recipe_fn,
    _count_dataset_tokens,
    _deduct_api_user_credits,
    run_recipe,
    get_torchtune_config_filename
)


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_config():
    with patch('torchtunewrapper.utils.config') as mock_cfg:
        mock_cfg.mock_user_has_enough_credits = False
        mock_cfg.customer_api_enabled = True
        mock_cfg.customer_api_url = "http://test-api.com"
        mock_cfg.customer_api_credits_deduct_endpoint = "/credits/deduct"
        mock_cfg.customer_api_key = "test-key"
        yield mock_cfg


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.__iter__ = MagicMock(return_value=iter([
        {'tokens': [1, 2, 3]},
        {'tokens': [4, 5, 6, 7]}
    ]))
    return dataset


def test_import_torchtune_recipe_fn_dummy():
    """Test importing dummy recipe"""
    recipe_fn = import_torchtune_recipe_fn(True, True, 'llm_dummy')
    assert recipe_fn.__module__ == 'torchtunewrapper.recipes.dummy'
    assert recipe_fn.__name__ == 'recipe_main'


def test_import_torchtune_recipe_fn_lora_single():
    """Test importing LoRA single device recipe"""
    recipe_fn = import_torchtune_recipe_fn(True, True, 'not_dummy')
    assert recipe_fn.__module__ == 'torchtunewrapper.recipes.lora_finetune_single_device'
    assert recipe_fn.__name__ == 'recipe_main'


def test_import_torchtune_recipe_fn_lora_distributed():
    """Test importing LoRA distributed recipe"""
    recipe_fn = import_torchtune_recipe_fn(True, False, 'not_dummy')
    assert recipe_fn.__module__ == 'torchtunewrapper.recipes.lora_finetune_distributed'
    assert recipe_fn.__name__ == 'recipe_main'


def test_import_torchtune_recipe_fn_full_single():
    """Test importing full finetune single device recipe"""
    recipe_fn = import_torchtune_recipe_fn(False, True, 'not_dummy')
    assert recipe_fn.__module__ == 'torchtunewrapper.recipes.full_finetune_single_device'
    assert recipe_fn.__name__ == 'recipe_main'


def test_import_torchtune_recipe_fn_full_distributed():
    """Test importing full finetune distributed recipe"""
    recipe_fn = import_torchtune_recipe_fn(False, False, 'not_dummy')
    assert recipe_fn.__module__ == 'torchtunewrapper.recipes.full_finetune_distributed'
    assert recipe_fn.__name__ == 'recipe_main'


def test_count_dataset_tokens(mock_dataset):
    """Test counting tokens in dataset"""
    count = _count_dataset_tokens(mock_dataset)
    assert count == 7  # Sum of token lengths: [1,2,3] and [4,5,6,7]


def test_deduct_api_user_credits_mock_enabled(mock_config, mock_logger):
    """Test credit deduction with mocking enabled"""
    mock_config.mock_user_has_enough_credits = True
    assert _deduct_api_user_credits("job1", "user1", 1000, 2, mock_logger)
    mock_logger.info.assert_called_with("Skipping credit check due to config settings")


def test_deduct_api_user_credits_api_disabled(mock_config, mock_logger):
    """Test credit deduction with API disabled"""
    mock_config.customer_api_enabled = False
    assert _deduct_api_user_credits("job1", "user1", 1000, 2, mock_logger)
    mock_logger.info.assert_called_with("Skipping credit check due to config settings")


def test_deduct_api_user_credits_system_user(mock_config, mock_logger):
    """Test credit deduction for system users"""
    system_users = ["0", "-1", "0x123"]
    for user in system_users:
        assert _deduct_api_user_credits("job1", user, 1000, 2, mock_logger)
        mock_logger.info.assert_called_with(f"Skipping credit check for user_id={user}")


@patch('requests.post')
def test_deduct_api_user_credits_success(mock_post, mock_config, mock_logger):
    """Test successful credit deduction API call"""
    mock_post.return_value.status_code = 200

    result = _deduct_api_user_credits("job1", "user1", 1000, 2, mock_logger)

    assert result is True
    mock_post.assert_called_once()
    args = mock_post.call_args
    assert args[0][0] == "http://test-api.com/credits/deduct"
    assert args[1]["headers"]["x-api-key"] == "test-key"
    assert args[1]["json"]["usage_amount"] == 2000  # 1000 tokens * 2 epochs


@patch('requests.post')
def test_deduct_api_user_credits_failure(mock_post, mock_config, mock_logger):
    """Test failed credit deduction API call"""
    mock_post.return_value.status_code = 400

    result = _deduct_api_user_credits("job1", "user1", 1000, 2, mock_logger)

    assert result is False
    mock_post.assert_called_once()


def test_run_recipe_insufficient_credits(mock_dataset, mock_logger):
    """Test run_recipe handling of insufficient credits"""
    mock_recipe = MagicMock()
    mock_cfg = DictConfig({"epochs": 1, "tokenizer": {"_component_": "torchtunewrapper.recipes.dummy.DummyTokenizer"}})

    with patch('torchtunewrapper.utils._deduct_api_user_credits', return_value=False), \
            pytest.raises(PermissionError, match="User does not have enough credits"):
        run_recipe(mock_recipe, "job1", "user1", mock_cfg, mock_dataset)


@patch('threading.Thread')
@patch('torchtunewrapper.utils._deduct_api_user_credits', return_value=True)
def test_run_recipe_success(mock_deduct, mock_thread, mock_dataset, mock_logger):
    """Test successful run_recipe execution"""
    mock_recipe = MagicMock()
    mock_recipe.return_value.total_epochs = 1
    mock_cfg = DictConfig({"epochs": 1, "tokenizer": {"_component_": "torchtunewrapper.recipes.dummy.DummyTokenizer"}})

    # Mock the recipe instance
    recipe_instance = MagicMock()
    mock_recipe.return_value = recipe_instance

    run_recipe(mock_recipe, "job1", "user1", mock_cfg, mock_dataset)

    # Verify setup and execution sequence
    recipe_instance.setup.assert_called_once()
    recipe_instance.train.assert_called_once()
    recipe_instance.save_checkpoint.assert_called_once()
    recipe_instance.cleanup.assert_called_once()


def test_get_torchtune_config_filename():
    """Test torchtune config filename generation"""
    test_cases = [
        {
            'model_base': 'hf://meta-llama/Meta-Llama-3.1-8B-Instruct',
            'use_lora': True,
            'use_qlora': False,
            'use_single_device': True,
            'expected': 'llama3_1/8B_lora_single_device.yml'
        },
        {
            'model_base': 'hf://meta-llama/Meta-Llama-3.1-70B-Instruct',
            'use_lora': False,
            'use_qlora': False,
            'use_single_device': False,
            'expected': 'llama3_1/70B_full.yml'
        },
    ]

    for case in test_cases:
        result = get_torchtune_config_filename(
            case['model_base'],
            case['use_lora'],
            case['use_qlora'],
            case['use_single_device']
        )
        assert result == case['expected']


def test_get_torchtune_config_filename_invalid_model():
    """Test handling of invalid model base"""
    with pytest.raises(ValueError, match="Unsupported model base"):
        get_torchtune_config_filename(
            'hf://invalid/model',
            True,
            False,
            True
        )
