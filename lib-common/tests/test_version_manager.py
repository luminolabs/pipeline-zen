import re
from unittest.mock import patch

import pytest
from io import StringIO
from common.version_manager import increment_version, VERSION_REGEX


def test_increment_version():
    initial_version = "0.5.0"
    with patch('common.version_manager.read_version', return_value=initial_version):
        with patch('common.version_manager.write_version', return_value=None):
            assert increment_version("patch") == "0.5.1"
            assert increment_version("minor") == "0.6.0"
            assert increment_version("major") == "1.0.0"

    with pytest.raises(ValueError):
        increment_version("something_else")


def test_semver_regex():
    assert re.match(VERSION_REGEX, "1.2.3")
    assert re.match(VERSION_REGEX, "10.0.15")
    assert not re.match(VERSION_REGEX, "1.2")
    assert not re.match(VERSION_REGEX, "v1.0.0")
