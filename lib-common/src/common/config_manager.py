import os

import yaml


class ConfigManager:
    """
    Manages loading application configurations from YAML files, with support for
    separate environments and overrides using environment variables.
    """

    def __init__(self):
        """
        Initializes the configuration manager.
        """
        # Configuration folder path
        self.base_path = os.environ.get('PZ_CONF_PATH', 'app-config')
        # Get the application environment
        self.current_env = os.environ.get('PZ_ENV', 'local')
        # Load configuration
        self.config = self.load()

    def load(self) -> dict:
        """
        Loads configuration from the base directory, with environment-specific overrides
        and the ability to override values with environment variables.

        :return: The merged configuration dictionary.
        """

        # Default configuration
        default_file = os.path.join(self.base_path, 'default.yml')
        # Environment specific configuration
        env_file = os.path.join(self.base_path, f'{self.current_env}.yml')

        config = {}
        # Load configuration from config files
        for file in (default_file, env_file):
            if os.path.exists(file):
                with open(file, 'r') as f:
                    config.update(yaml.safe_load(f))

        # Override configuration with environment variables
        for key, value in config.items():
            env_var_name = f'PZ_{key.upper()}'
            if env_var_name in os.environ:
                env_var_value = os.environ[env_var_name]
                if is_truthy(env_var_value):
                    env_var_value = True
                elif is_falsy(env_var_value):
                    env_var_value = False
                config[key] = env_var_value

        return config

    def __getattr__(self, attr):
        return self.config[attr]


def is_truthy(value) -> bool:
    """
    Checks if the given value is truthy
    :param value: Value to check
    :return: True if the given value is truthy
    """
    if isinstance(value, str):
        value = value.lower()
    return any(value == x for x in (True, 'true', '1', 'yes'))


def is_falsy(value) -> bool:
    """
    Checks if the given value is falsy
    :param value: Value to check
    :return: True if the given value is falsy
    """
    if isinstance(value, str):
        value = value.lower()
    return any(value == x for x in (False, 'false', '0', 'no'))


config = ConfigManager()
