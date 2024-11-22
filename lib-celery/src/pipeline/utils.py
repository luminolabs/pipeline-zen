from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any, Tuple, Callable, Dict

from celery import Celery, chain
from celery.signals import task_failure

from common.agent.system_specs import SystemSpecsAgent
from common.utils import setup_logger


@dataclass
class BaseWorkflowConfig:
    """Base class for workflow configuration"""
    job_id: str
    user_id: str
    device: str
    celery_broker_url: str
    args: Tuple
    env: str
    wf_name: str


class TaskRegistry:
    """Registry for Celery tasks"""

    def __init__(self):
        self.app = None

        # Link the function name to the function itself, so that we can retrieve the wrapped function later.
        # Initially, the function name is the key and the function is the value.
        # After the task is registered with the Celery app, the task is the value.
        # The task is a Celery task object, which is a wrapper around the function.
        self.tasks: Dict[str, Callable] = {}

    def set_app_and_register_tasks(self, app: Celery):
        """
        Set the app for the task registry, and register all tasks with the app
        """
        self.app = app
        for task_name, task in self.tasks.items():
            self.tasks[task_name] = self.app.task(task)

    def add_task(self, func):
        """
        Adds a task to the list so it can be registered with the Celery app later
        """
        self.tasks[func.__name__] = func
        return func


class BaseWorkflowManager:
    """Manages workflow execution and task scheduling"""

    def __init__(self, config: BaseWorkflowConfig, task_registry: TaskRegistry):
        # Set object attributes
        self.config = config
        self.task_registry = task_registry
        # Set up the logger
        self.logger = setup_logger(f'celery_{config.wf_name}_wf', config.job_id, config.user_id)
        # Create the Celery app and register the failure handler
        self.app = self._create_celery_app()
        self._register_failure_handler()

    def _create_celery_app(self) -> Celery:
        """Creates and configures the Celery app"""
        app = Celery(self.config.wf_name, broker=self.config.celery_broker_url)
        return app

    def _register_failure_handler(self):
        """Registers the failure handler for Celery tasks"""

        @task_failure.connect
        def handle_task_failure(*args, **kwargs):
            self.logger.error('Something went wrong during task execution')
            self.app.control.shutdown()

    @abstractmethod
    def _create_tasks(self) -> List[Any]:
        raise NotImplementedError("This method must be implemented in a subclass.")

    def schedule(self):
        """Schedules the workflow execution"""
        if self.config.env != 'local' and self.config.device == 'cuda':
            system_specs = SystemSpecsAgent(self.logger)
            if system_specs.get_gpu_spec() is None:
                raise RuntimeError('No GPUs found on this machine')

        tasks = self._create_tasks()
        chain(*tasks)()

    def start_worker(self):
        """Starts the Celery worker"""
        argv = ['worker', '--loglevel=INFO', '--pool=solo']
        self.app.worker_main(argv)


def schedule(workflow_manager: BaseWorkflowManager):
    """Schedules the workflow tasks for execution later"""
    workflow_manager.schedule()


def start_worker(workflow_manager: BaseWorkflowManager):
    """Starts the celery worker which will execute the workflow tasks"""
    workflow_manager.start_worker()
