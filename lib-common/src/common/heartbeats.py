import functools
from typing import Optional

from common.config_manager import config
from common.gcp import get_vm_name_from_metadata, publish_to_pubsub
from common.utils import utcnow, utcnow_str, job_meta_context


def heartbeat_wrapper(workflow_name, task_name):
    """
    A decorator that sends a heartbeat message to the pipeline-zen-jobs-heartbeats topic
    when a task starts, finishes, or errors out.

    :param workflow_name: The name of the workflow
    :param task_name: The name of the task
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            job_id = config.job_id
            user_id = config.user_id
            start_time = utcnow()
            # Send a start heartbeat when the task starts
            send_heartbeat(job_id, user_id, f"wf-{workflow_name}-{task_name}-start")
            try:
                result = func(*args, **kwargs)
                if result is not False:
                    # Function run successfully
                    send_heartbeat(job_id, user_id, f"wf-{workflow_name}-{task_name}-finish")
                else:
                    # Function returned False, indicating an error, the error is already logged by the function
                    send_heartbeat(job_id, user_id, f"wf-{workflow_name}-{task_name}-error")
            except Exception as e:
                send_heartbeat(job_id, user_id, f"wf-{workflow_name}-{task_name}-error")
                raise e
            # Send a total heartbeat with the elapsed time
            send_heartbeat(
                job_id, user_id, f"wf-{workflow_name}-{task_name}-total",
                elapsed_time=(utcnow() - start_time).total_seconds())
            return result

        return wrapper

    return decorator


def send_heartbeat(job_id: str, user_id: str, status: str, elapsed_time: Optional[float] = None):
    """
    Send a heartbeat message to the pipeline-zen-jobs-heartbeats topic.

    :param job_id: The job id
    :param user_id: The user id
    :param status: The status of the job
    :param elapsed_time: The elapsed time of the job in seconds
    """
    msg = {'status': status, 'vm_name': get_vm_name_from_metadata(),
           'timestamp': utcnow_str(),
           'elapsed_time_s': f'{elapsed_time:.2f}' if elapsed_time else None}
    with job_meta_context(job_id, user_id) as job_meta:
        job_meta.setdefault('heartbeats', []).append(msg)
    publish_to_pubsub(job_id, user_id, config.heartbeat_topic, msg)
