import os
from unittest.mock import patch

from common.config_manager import ConfigManager


def test_load_default():
    config_manager = ConfigManager()
    config = config_manager.config
    assert config['root_path'] == '.'


def test_load_environment_override():
    with patch.dict(os.environ, {'PZ_ENV': 'test'}):
        config_manager = ConfigManager()
        config = config_manager.config
        assert config['test_config'] == 'test_config'


def test_environment_variable_override():
    with patch.dict(os.environ, {'PZ_ROOT_PATH': 'env_override'}):
        config_manager = ConfigManager()
        config = config_manager.config
        assert config['root_path'] == 'env_override'
