import re

VERSION_REGEX = r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"


def read_version() -> str:
    """
    Reads the current version from the 'VERSION' file.

    :return: The current version as a string.
    :raises ValueError: if the version string is invalid.
    """
    with open('VERSION', 'r') as f:
        version = f.read().strip()
        if not re.match(VERSION_REGEX, version):
            raise ValueError('Invalid version format in VERSION file; '
                             'must be X.Y.Z, where X, Y and Z are integers.')
        return version


def write_version(version: str) -> None:
    """
    Writes the new version to the 'VERSION' file.

    :param version: The version to write to the 'VERSION' file.
    """
    with open('VERSION', 'w') as f:
        f.write(version)


def increment_version(part: str = 'patch') -> str:
    """
    Increments the specified part of the version (`major`, `minor`, `patch`).

    :param part: The part of the version to increment.
        Must be one of `major`, `minor`, or `patch`. Defaults to `patch`.

    :return: The incremented version as a string.
    :raises ValueError: if the version string is invalid.
    """
    current_version = read_version()
    version_parts = re.match(VERSION_REGEX, current_version).groupdict()

    if part == 'major':
        version_parts['major'] = str(int(version_parts['major']) + 1)
        version_parts['minor'] = '0'
        version_parts['patch'] = '0'
    elif part == 'minor':
        version_parts['minor'] = str(int(version_parts['minor']) + 1)
        version_parts['patch'] = '0'
    elif part == 'patch':
        version_parts['patch'] = str(int(version_parts['patch']) + 1)
    else:
        raise ValueError('Invalid part: must be `major`, `minor`, or `patch`')

    new_version = '{major}.{minor}.{patch}'.format(**version_parts)
    write_version(new_version)
    return new_version
