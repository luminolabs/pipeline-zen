#!python

import argparse
import logging
from common.version_manager import increment_version

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Bump application version')
    parser.add_argument('-p', '--part', type=str, required=False,
                        default='patch',
                        help='The version part to increment. '
                             'Must be one of `major`, `minor`, or `patch`. Defaults to `patch`.')
    args = parser.parse_args()
    part = args.part

    logging.info(f'Starting the version bump script; updating part {part}')
    # Increment the version
    new_version = increment_version(part)
    logging.info(f'Version incremented to {new_version}')
