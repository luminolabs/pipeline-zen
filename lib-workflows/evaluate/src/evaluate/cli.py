import argparse

from evaluate.workflow import main


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run the model evaluation workflow")
    add_parser_args(parser)
    args = parser.parse_args()

    return args.job_id, args.user_id, args.job_config_name, args.batch_size, args.num_batches


def add_parser_args(parser: argparse.ArgumentParser):
    """
    Add command line arguments to the parser.
    Allows for reuse of the argument parsing logic in other scripts.

    :param parser: The parser to add arguments to
    :return: None
    """
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help='The name of the job config file, without the `.py` extension')
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help='The job_id to use for evaluation; '
                             'logs and other job results and artifacts will be named after this.')
    parser.add_argument('-uid', '--user_id', type=str, required=False, default='0',
                        help="The user_id to use for training; "
                             "logs and other job results and artifacts will be named after this.")

    parser.add_argument('-bs', '--batch_size', type=int, required=False,
                        help='The batch size to use for training ')
    parser.add_argument('-nb', '--num_batches', type=int, required=False,
                        help='The number of batches to run. This is helpful when testing code changes;'
                             'the trainer will stop after this many batches, and continue to the'
                             'next epoch')


if __name__ == '__main__':
    # Run evaluate workflow
    main(*parse_args())
