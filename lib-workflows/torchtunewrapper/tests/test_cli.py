import argparse
from unittest.mock import patch, MagicMock

import pytest

from torchtunewrapper.cli import parse_args, add_parser_args


@pytest.fixture
def mock_config():
    with patch('torchtunewrapper.cli.config') as mock_cfg:
        yield mock_cfg


@pytest.fixture
def mock_argparse():
    with patch('torchtunewrapper.cli.argparse.ArgumentParser') as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        yield mock_parser


def test_parse_args_required_args(mock_config):
    """Test parsing with only required arguments"""
    test_args = [
        "--job_config_name", "test_config",
        "--dataset_id", "test_dataset",
        "--num_gpus", "2"
    ]

    with patch('sys.argv', ['script.py'] + test_args):
        result = parse_args()

    assert result == (None, "0", "test_config", "test_dataset", 1, True, 1, True, False, 3e-4, None, 2, None)
    mock_config.set.assert_called_with('user_id', "0")


def test_parse_args_all_args(mock_config):
    """Test parsing with all possible arguments"""
    test_args = [
        "--job_config_name", "test_config",
        "--job_id", "test_job",
        "--user_id", "test_user",
        "--dataset_id", "test_dataset",
        "--batch_size", "4",
        "--shuffle", "false",
        "--num_epochs", "3",
        "--use_lora", "false",
        "--use_qlora", "true",
        "--lr", "0.001",
        "--seed", "42",
        "--num_gpus", "2",
        "--pytorch_cuda_alloc_conf", "max_split_size_mb:512"
    ]

    with patch('sys.argv', ['script.py'] + test_args):
        result = parse_args()

    assert result == ("test_job", "test_user", "test_config", "test_dataset", 4, False, 3,
                      False, True, 0.001, 42, 2, "max_split_size_mb:512")
    mock_config.set.assert_called_with('user_id', "test_user")


def test_parse_args_boolean_conversion():
    """Test boolean argument conversion"""
    test_cases = [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False)
    ]

    for input_val, expected in test_cases:
        test_args = [
            "--job_config_name", "test_config",
            "--dataset_id", "test_dataset",
            "--num_gpus", "2",
            "--shuffle", input_val,
            "--use_lora", input_val
        ]

        with patch('sys.argv', ['script.py'] + test_args):
            result = parse_args()
            assert result[5] == expected  # shuffle
            assert result[7] == expected  # use_lora


def test_add_parser_args():
    """Test argument addition to parser"""
    parser = argparse.ArgumentParser()
    add_parser_args(parser)

    # Convert parser arguments to dict for easier testing
    actions = {action.dest: action for action in parser._actions}

    # Test required arguments
    assert actions['job_config_name'].required
    assert actions['dataset_id'].required
    assert actions['num_gpus'].required

    # Test default values
    assert actions['batch_size'].default == 1
    assert actions['shuffle'].default == True
    assert actions['num_epochs'].default == 1
    assert actions['use_lora'].default == True
    assert actions['use_qlora'].default == False
    assert actions['lr'].default == 3e-4
    assert actions['user_id'].default == "0"


def test_parse_args_invalid_args():
    """Test handling of invalid arguments"""
    test_args = [
        "--job_config_name", "test_config",
        "--dataset_id", "test_dataset",
        "--num_gpus", "invalid"  # Invalid int value
    ]

    with patch('sys.argv', ['script.py'] + test_args), \
            pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("required_arg", ["job_config_name", "dataset_id", "num_gpus"])
def test_parse_args_missing_required(required_arg):
    """Test handling of missing required arguments"""
    test_args = [
        "--job_config_name", "test_config",
        "--dataset_id", "test_dataset",
        "--num_gpus", "2"
    ]

    # Remove the required argument we want to test
    arg_index = test_args.index(f"--{required_arg}")
    test_args.pop(arg_index + 1)
    test_args.pop(arg_index)

    with patch('sys.argv', ['script.py'] + test_args), \
            pytest.raises(SystemExit):
        parse_args()


def test_parse_args_updates_config(mock_config):
    """Test that parse_args updates config correctly"""
    test_args = [
        "--job_config_name", "test_config",
        "--job_id", "test_job",
        "--user_id", "test_user",
        "--dataset_id", "test_dataset",
        "--num_gpus", "2"
    ]

    with patch('sys.argv', ['script.py'] + test_args):
        parse_args()

    assert mock_config.set.call_count == 2
    mock_config.set.assert_any_call('job_id', 'test_job')
    mock_config.set.assert_any_call('user_id', 'test_user')
