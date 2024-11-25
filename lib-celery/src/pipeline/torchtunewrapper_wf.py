import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Any

from common.config_manager import config
from common.gcp import get_results_bucket, make_object_public, upload_directory
from common.heartbeats import heartbeat_wrapper
from common.utils import get_work_dir, setup_logger, get_artifacts
from pipeline.utils import BaseWorkflowConfig, schedule, start_worker, TaskRegistry, BaseWorkflowManager
from torchtunewrapper.cli import parse_args as torchtunewrapper_parse_args
from torchtunewrapper.workflow import main as torchtunewrapper_


@dataclass
class WorkflowConfig(BaseWorkflowConfig):
    """Configuration for the workflow execution"""
    delete_results: bool
    send_to_gcs: bool
    root_path: str
    work_dir: str


class WorkflowManager(BaseWorkflowManager):
    """Manages workflow execution and task scheduling"""

    # Set type hint
    config: WorkflowConfig

    def _create_tasks(self) -> List[Any]:
        """Creates the task chain based on configuration"""

        # Get the job_id, user_id, and other arguments from the config
        job_id = self.config.job_id
        user_id = self.config.user_id
        args = self.config.args

        # Get the tasks from the task registry
        _mark_started = self.task_registry.tasks['mark_started']
        _torchtunewrapper = self.task_registry.tasks['torchtunewrapper']
        _mark_finished = self.task_registry.tasks['mark_finished']
        _upload_results = self.task_registry.tasks['upload_results']
        _delete_results = self.task_registry.tasks['delete_results']
        _shutdown_celery_worker = self.task_registry.tasks['shutdown_celery_worker']

        # Create the task chain
        tasks = [_mark_started.s(None, job_id, user_id),
                 _torchtunewrapper.s(*args),
                 _mark_finished.s(job_id, user_id)]
        if self.config.send_to_gcs:
            tasks.append(_upload_results.s(job_id, user_id))
        if self.config.delete_results:
            tasks.append(_delete_results.s(job_id, user_id))
        tasks.append(_shutdown_celery_worker.s(job_id, user_id))

        # Return the task chain
        return tasks


# Initialize the workflow manager and task registry
task_registry = TaskRegistry()


@task_registry.add_task
def torchtunewrapper(_, job_id: str, user_id: str, job_config_name: str,
                     dataset_id: Optional[str] = None,
                     batch_size: int = 1, shuffle: bool = True, num_epochs: int = 1,
                     use_lora: bool = True, use_qlora: bool = False,
                     lr: float = 3e-4, seed: Optional[int] = None,
                     num_gpus: int = 1,
                     pytorch_cuda_alloc_conf: str = None):
    logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
    try:
        return torchtunewrapper_(
            job_id, user_id, job_config_name,
            dataset_id,
            batch_size, shuffle, num_epochs,
            use_lora, use_qlora,
            lr, seed,
            num_gpus,
            pytorch_cuda_alloc_conf)
    except Exception as e:
        logger.error(f'`torchtunewrapper` task failed with error: {e}\n{traceback.format_exc()}')
        return False


@heartbeat_wrapper("torchtunewrapper", "upload_weights")
@task_registry.add_task
def upload_results(mark_finished_result: bool, job_id: str, user_id: str):
    """Upload results and logs to Google Cloud Storage."""
    if not mark_finished_result:
        return

    work_dir = get_work_dir(job_id, user_id)
    results_bucket_name = get_results_bucket()
    upload_directory(work_dir, results_bucket_name)

    weight_files, other_files = get_artifacts(job_id, user_id)
    for f in weight_files + other_files:
        make_object_public(results_bucket_name, f'{user_id}/{job_id}/{f}')


@task_registry.add_task
def delete_results(_, job_id: str, user_id: str):
    """Deletes the job results and logs directories."""
    work_dir = get_work_dir(job_id, user_id)
    os.system(f'rm -rf {work_dir}')


@task_registry.add_task
def mark_finished(torchtunewrapper_result: bool, job_id: str, user_id: str):
    """Creates a `.finished` file to signal job completion."""
    logger = setup_logger('celery_torchtunewrapper_wf', job_id, user_id)
    if not torchtunewrapper_result:
        logger.warning(f'`torchtunewrapper` task failed - will not run `mark_finished` task')
        return False

    work_dir = get_work_dir(job_id, user_id)
    path = os.path.join(work_dir, config.finished_file)
    with open(path, "w") as f:
        f.write(job_id + "\n")
    return True


@task_registry.add_task
def mark_started(_, job_id: str, user_id: str):
    """Creates a `.started` file to signal job start."""
    path = os.path.join(get_work_dir(job_id, user_id), config.started_file)
    with open(path, "w") as f:
        f.write(job_id + "\n")


@task_registry.add_task
def shutdown_celery_worker(_, job_id: str, user_id: str):
    """
    Shuts down the celery worker.
    Note: Don't implement unit tests for this task.
    """
    workflow_manager.app.control.shutdown()


# Entry point for the Celery worker
if __name__ == '__main__':
    """Initializes the workflow manager and task registry"""
    # Parse command line arguments
    args = torchtunewrapper_parse_args()

    # Set the job_id and user_id in the config
    config.set('job_id', args[0])
    config.set('user_id', args[1])

    # Initialize the workflow manager
    workflow_config = WorkflowConfig(
        job_id=config.job_id,
        user_id=config.user_id,
        device=config.device,
        celery_broker_url=config.celery_broker_url,
        args=args,
        env=config.env,
        wf_name='torchtunewrapper',
        delete_results=config.delete_results,
        send_to_gcs=config.send_to_gcs,
        root_path=config.root_path,
        work_dir=config.work_dir,
    )
    workflow_manager = WorkflowManager(workflow_config, task_registry)

    # Set the app and register the tasks
    task_registry.set_app_and_register_tasks(workflow_manager.app)

    # Schedule the workflow and start the worker
    schedule(workflow_manager)
    start_worker(workflow_manager)
