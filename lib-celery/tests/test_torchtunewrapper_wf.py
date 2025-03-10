import os
from unittest.mock import patch, MagicMock

import pytest
from celery import Celery

from common.config_manager import config
from pipeline.torchtunewrapper_wf import (
    WorkflowConfig,
    WorkflowManager,
    upload_results,
    delete_results,
    mark_finished,
    mark_started,
    torchtunewrapper,
    TaskRegistry,
)

# Test Data
TEST_JOB_ID = "test-job-123"
TEST_USER_ID = "test-user-456"
TEST_ARGS = (TEST_JOB_ID, TEST_USER_ID, "config-name", "dataset-id", 2, True, 1, True, False, 0.001, None, 1)


@pytest.fixture
def mock_config(monkeypatch):
    """Mock the config singleton's job_id and user_id"""
    monkeypatch.setattr('common.config_manager.config.job_id', TEST_JOB_ID)
    monkeypatch.setattr('common.config_manager.config.user_id', TEST_USER_ID)


@pytest.fixture
def mock_celery_app():
    app = MagicMock(spec=Celery)
    app.control = MagicMock()
    app.control.shutdown = MagicMock()
    return app


@pytest.fixture
def workflow_config():
    return WorkflowConfig(
        job_id=TEST_JOB_ID,
        user_id=TEST_USER_ID,
        device="cuda",
        celery_broker_url="memory://localhost/",
        delete_results=True,
        send_to_gcs=True,
        root_path=".",
        work_dir=".results",
        wf_name="torchtunewrapper",
        env="local",
        args=TEST_ARGS
    )


@pytest.fixture
def task_registry(mock_celery_app):
    registry = TaskRegistry()
    registry.app = mock_celery_app
    return registry


# TaskRegistry Tests
def test_task_registry_add_task(task_registry):
    @task_registry.add_task
    def test_task():
        pass

    assert "test_task" in task_registry.tasks
    assert callable(task_registry.tasks["test_task"])


def test_task_registry_set_app_and_register_tasks(mock_celery_app):
    registry = TaskRegistry()

    @registry.add_task
    def test_task():
        pass

    # registry.add_task(test_task)
    registry.set_app_and_register_tasks(mock_celery_app)
    assert registry.app == mock_celery_app
    mock_celery_app.task.assert_called_once_with(test_task)


# WorkflowManager Tests
def test_workflow_manager_initialization(workflow_config, task_registry):
    manager = WorkflowManager(workflow_config, task_registry)
    assert manager.config == workflow_config
    assert isinstance(manager.app, Celery)
    assert manager.task_registry == task_registry


@patch("pipeline.utils.SystemSpecsAgent")
def test_workflow_manager_schedule_with_gpu(mock_system_specs, workflow_config, task_registry):
    mock_system_specs.return_value.get_gpu_spec.return_value = {"gpu": "present"}
    manager = WorkflowManager(workflow_config, task_registry)
    manager._create_tasks = MagicMock()

    with patch("pipeline.utils.chain") as mock_chain:
        manager.schedule()
        mock_chain.assert_called_once()


@patch("pipeline.utils.SystemSpecsAgent")
def test_workflow_manager_schedule_no_gpu_prod(mock_system_specs, workflow_config, task_registry):
    mock_system_specs.return_value.get_gpu_spec.return_value = None
    workflow_config.env = "prod"  # Non-local env
    manager = WorkflowManager(workflow_config, task_registry)

    with pytest.raises(RuntimeError, match="No GPUs found on this machine"):
        manager.schedule()


def test_workflow_manager_start_worker(workflow_config, task_registry):
    manager = WorkflowManager(workflow_config, task_registry)

    with patch.object(manager.app, "worker_main") as mock_worker_main:
        manager.start_worker()
        mock_worker_main.assert_called_once_with(["worker", "--loglevel=INFO", "--pool=solo"])


# Task Tests
@pytest.mark.parametrize("tmp_path_str", ["/tmp/test"])
@patch("pipeline.torchtunewrapper_wf.get_work_dir")
def test_mark_started(mock_get_work_dir, tmp_path_str, tmp_path):
    mock_get_work_dir.return_value = tmp_path_str
    mark_started(None, TEST_JOB_ID, TEST_USER_ID)

    # Create the directory since it's mocked
    os.makedirs(tmp_path_str, exist_ok=True)
    started_file = os.path.join(tmp_path_str, config.started_file)

    with open(started_file, 'r') as f:
        content = f.read().strip()
        assert content == TEST_JOB_ID


@pytest.mark.parametrize("tmp_path_str", ["/tmp/test"])
@patch("pipeline.torchtunewrapper_wf.get_work_dir")
def test_mark_finished_success(mock_get_work_dir, tmp_path_str):
    mock_get_work_dir.return_value = tmp_path_str
    os.makedirs(tmp_path_str, exist_ok=True)

    result = mark_finished(True, TEST_JOB_ID, TEST_USER_ID)
    assert result is True

    finished_file = os.path.join(tmp_path_str, config.finished_file)
    with open(finished_file, 'r') as f:
        content = f.read().strip()
        assert content == TEST_JOB_ID


@patch("pipeline.torchtunewrapper_wf.get_work_dir")
def test_mark_finished_failure(mock_get_work_dir, tmp_path):
    mock_get_work_dir.return_value = str(tmp_path)
    result = mark_finished(False, TEST_JOB_ID, TEST_USER_ID)

    assert result is False
    finished_file = tmp_path / config.finished_file
    assert not finished_file.exists()


@patch("pipeline.torchtunewrapper_wf.get_work_dir")
def test_delete_results(mock_get_work_dir, tmp_path):
    work_dir = tmp_path / TEST_USER_ID / TEST_JOB_ID
    epoch_dir = work_dir / "epoch_0" / "weights.safetensors"
    epoch_dir.mkdir(parents=True)

    test_file = work_dir / "test.txt"
    test_file.write_text("test content")

    mock_get_work_dir.return_value = str(work_dir)
    delete_results(None, TEST_JOB_ID, TEST_USER_ID)

    assert work_dir.exists()  # Work dir should still exist so that logs can be retrieved
    assert not epoch_dir.exists()  # Epoch dir should be deleted to save space on the worker


@patch("pipeline.torchtunewrapper_wf.upload_directory")
@patch("pipeline.torchtunewrapper_wf.get_artifacts")
@patch("pipeline.torchtunewrapper_wf.make_object_public")
def test_upload_results_success(mock_make_public, mock_get_artifacts, mock_upload, mock_config):
    mock_get_artifacts.return_value = (["weights.pt"], ["config.json"])

    upload_results(True, TEST_JOB_ID, TEST_USER_ID)

    mock_upload.assert_called_once()
    assert mock_make_public.call_count == 2


def test_upload_results_skip_on_failure(mock_config):
    result = upload_results(False, TEST_JOB_ID, TEST_USER_ID)
    assert result is None


@patch("pipeline.torchtunewrapper_wf.torchtunewrapper_")
def test_torchtunewrapper_success(mock_torchtune, mock_config):
    mock_torchtune.return_value = True
    result = torchtunewrapper(None, TEST_JOB_ID, TEST_USER_ID, "config-name")

    assert result is True
    mock_torchtune.assert_called_once()


@patch("pipeline.torchtunewrapper_wf.torchtunewrapper_")
def test_torchtunewrapper_error_handling(mock_torchtune, mock_config):
    mock_torchtune.side_effect = Exception("Test error")
    result = torchtunewrapper(None, TEST_JOB_ID, TEST_USER_ID, "config-name")

    assert result is False


# Integration Tests
@pytest.mark.asyncio
@patch("pipeline.torchtunewrapper_wf.get_work_dir")
async def test_workflow_execution(mock_get_work_dir, tmp_path):
    work_dir = tmp_path / TEST_USER_ID / TEST_JOB_ID
    work_dir.mkdir(parents=True)
    mock_get_work_dir.return_value = str(work_dir)

    # Execute tasks in sequence
    mark_started(None, TEST_JOB_ID, TEST_USER_ID)
    torchtune_result = True  # Simulate successful training
    mark_finished(torchtune_result, TEST_JOB_ID, TEST_USER_ID)

    # Verify workflow artifacts
    assert (work_dir / config.started_file).exists()
    assert (work_dir / config.finished_file).exists()


# Configuration Tests
def test_workflow_config_validation(workflow_config):
    assert workflow_config.job_id == TEST_JOB_ID
    assert workflow_config.user_id == TEST_USER_ID
    assert workflow_config.device == "cuda"
    assert workflow_config.delete_results is True
    assert workflow_config.send_to_gcs is True
