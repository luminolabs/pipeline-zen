import argparse

from evaluate.workflow import main


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model evaluation workflow")
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help='The name of the job config file, without the `.py` extension')
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help='The job_id to use for evaluation; '
                             'logs and other job results and artifacts will be named after this.')
    parser.add_argument('-bs', '--batch_size', type=int, required=False,
                        help='The batch size to use for training ')
    parser.add_argument('-nb', '--num_batches', type=int, required=False,
                        help='The number of batches to run. This is helpful when testing code changes;'
                             'the trainer will stop after this many batches, and continue to the'
                             'next epoch')
    args = parser.parse_args()

    return args.job_config_name, args.job_id, args.batch_size, args.num_batches


if __name__ == '__main__':
    # Run evaluate workflow
    main(*parse_args())
