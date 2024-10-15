import os
from typing import Tuple, List

from common.utils import get_work_dir


def get_artifacts(job_id: str, user_id: str) -> Tuple[List[str], List[str]]:
    """
    Get the artifacts for a given job

    :param job_id: The job id
    :param user_id: The user id
    :return: A dictionary of artifacts
    """
    work_dir = get_work_dir(job_id, user_id)
    weight_files = [f for f in os.listdir(work_dir) if f.endswith('.pt')]
    other_files = [f for f in os.listdir(work_dir) if f in ['config.json']]
    return weight_files, other_files
