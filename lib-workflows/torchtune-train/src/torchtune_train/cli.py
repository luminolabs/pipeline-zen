import argparse

from torchtune_train.workflow import main


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model training workflow")
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help="The name of the job config file, without the `.py` extension")
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help="The job_id to use for training; "
                             "logs and other job results and artifacts will be named after this.")
    args = parser.parse_args()

    return args.job_config_name, args.job_id


if __name__ == '__main__':
    # Run train workflow
    main(*parse_args())
