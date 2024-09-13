import argparse

from common.config_manager import is_truthy, config
from torchtunewrapper.workflow import main


def parse_args() -> tuple:
    """
    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="The model training workflow")
    add_parser_args(parser)
    args = parser.parse_args()

    # Update the application config so that they can be accessed globally
    config.set('job_id', args.job_id)
    config.set('user_id', args.user_id)

    return (args.job_id, args.user_id, args.job_config_name,
            args.dataset_id, args.train_file_path,
            args.batch_size, args.shuffle, args.num_epochs,
            args.use_lora, args.use_qlora, args.num_gpus,
            args.pytorch_cuda_alloc_conf)


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
    parser.add_argument('-uid', '--user_id', type=str, required=False, default='0',
                        help="The user_id to use for training; "
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
    parser.add_argument('-ql', '--use_qlora', type=is_truthy, required=False, default=False,
                        help="Whether to use the QLoRA; default is False and only used if `use_lora` is True")

    parser.add_argument('-gpus', '--num_gpus', type=int, required=True, default=1,
                        help="The number of GPUs to use for training; default is 1")

    parser.add_argument('-pca', '--pytorch_cuda_alloc_conf', type=str, required=False,
                        help="The PyTorch CUDA allocation configuration; default is `expandable_segments:True`")


if __name__ == '__main__':
    # Run torchtune workflow
    main(*parse_args())
