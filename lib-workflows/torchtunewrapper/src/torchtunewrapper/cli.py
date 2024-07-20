import argparse

from common.config_manager import is_truthy
from torchtunewrapper.workflow import main


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model training workflow")
    add_parser_args(parser)
    args = parser.parse_args()

    return (args.job_config_name, args.job_id,
            args.dataset_id, args.train_file_path,
            args.batch_size, args.shuffle, args.num_epochs,
            args.use_lora,
            args.use_single_device, args.num_gpus)


def add_parser_args(parser: argparse.ArgumentParser):
    """
    Add command line arguments to the parser.
    Allows for reuse of the argument parsing logic in other scripts.

    :param parser: The parser to add arguments to
    :return: None
    """
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help="The name of the job config file, without the `.py` extension")
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help="The job_id to use for training; "
                             "logs and other job results and artifacts will be named after this.")

    parser.add_argument('-ds', '--dataset_id', type=str, required=True,
                        help="The dataset repo to use for training; e.g. `tatsu-lab/alpaca`")
    parser.add_argument('-tfp', '--train_file_path', type=str, required=False,
                        help="The path to the train file in the repo to use for training; e.g. `train.jsonl`")
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=1,
                        help="The batch size to use for training; default is 1")
    parser.add_argument('-s', '--shuffle', type=is_truthy, required=False, default=True,
                        help="Whether to shuffle the training data; default is True")

    parser.add_argument('-ne', '--num_epochs', type=int, required=False, default=1,
                        help="The number of epochs to train for; default is 1")
    parser.add_argument('-l', '--use_lora', type=is_truthy, required=False, default=True,
                        help="Whether to use the LoRA; default is True")
    parser.add_argument('-sd', '--use_single_device', type=is_truthy, required=False, default=True,
                        help="Whether to use a single GPU device; default is True")
    parser.add_argument('-gpus', '--num_gpus', type=int, required=True, default=1,
                        help="The number of GPUs to use for training; default is 1")


if __name__ == '__main__':
    # Run torchtune workflow
    main(*parse_args())
