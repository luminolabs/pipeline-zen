#!python

import argparse

from common.version_manager import increment_version


def parse_args() -> str:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Bump application version")
    parser.add_argument('-p', '--part', nargs=1, type=str, required=False,
                        default='patch',
                        help="The version part to increment. "
                             "Must be one of `major`, `minor`, or `patch`. Defaults to `patch`.")
    args = parser.parse_args()
    part = args.part and args.part[0] if isinstance(args.part, list) else args.part
    return part


if __name__ == '__main__':
    part = parse_args()
    increment_version(part)
