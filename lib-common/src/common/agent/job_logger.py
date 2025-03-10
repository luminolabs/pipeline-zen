import json
from abc import abstractmethod
from copy import deepcopy
from logging import Logger
from typing import Optional, Union

from common.agent.system_specs import SystemSpecsAgent
from common.config_manager import config
from common.gcp import publish_to_pubsub, insert_to_biqquery, BIGQUERY_TIMESTAMP_FORMAT, get_results_bucket
from common.utils import AutoJSONEncoder, utcnow, utcnow_str, job_meta_context


class BaseJobLoggerAgent:
    """
    Base class for distributing logs across multiple channels
    """

    def __init__(self, job_id: str, user_id: str, agent_logger: Logger, main_logger: Logger):
        """
        :param job_id: The id of the job
        :param user_id: The id of the user
        :param agent_logger: The logger to use
        :param main_logger: The main logger to use
        """
        self._job_id = job_id
        self._user_id = user_id
        self._agent_logger = agent_logger
        self._main_logger = main_logger
        self._time_start = None
        self._time_end = None

    @property
    @abstractmethod
    def _workflow_name(self) -> str:
        pass

    def mark_time_start(self):
        """
        Mark, and log the start time of the training job
        """
        self._time_start = utcnow()
        self._log_data(operation='time_start')

    def mark_time_end(self):
        """
        Mark, and log the end time of the training job
        """
        self._time_end = utcnow()
        self._log_data(operation='time_end')

    def log_time_elapsed(self):
        """
        Log the training time length in minutes
        """
        time_delta_m = f'{(self._time_end - self._time_start).total_seconds() / 60:.4f} minutes'
        self._log_data(operation='time_elapsed', data=time_delta_m)

    def log_system_specs(self):
        """
        Log system specs
        """
        system_specs = SystemSpecsAgent(self._main_logger).get_specs()
        self._log_data(operation='system_specs', data=system_specs)

    def log_job_config(self, job_config: dict):
        """
        Log job configuration

        :param job_config: The job configuration
        """
        self._log_data(operation='job_config', data=job_config)

    def _log_data(self, operation: str, data: Optional[Union[dict, str]] = None):
        """
        Log data

        :param operation: The operation name of the data to be logged
        :param data: The result of the operation (optional)
        :return:
        """
        # Construct row
        row = {'job_id': self._job_id,
               'user_id': self._user_id,
               'create_ts': utcnow_str(),
               'workflow': self._workflow_name,
               'operation': operation,
               'data': data}

        # Push row to channels
        self._agent_logger.info(row)
        self._bq_insert(row)
        self._pubsub_send(row)
        self._file_write(row)

    def _file_write(self, row: dict):
        """
        Write scores to a file

        :param row: The row to write
        """
        with job_meta_context(self._job_id, self._user_id) as job_meta:
            job_meta.setdefault('job_logger', []).append(row)

    def _pubsub_send(self, row: dict):
        """
        Send scores to Pub/Sub

        :param row: The row to send
        """
        publish_to_pubsub(self._job_id, self._user_id,
                          topic_name=config.jobs_meta_topic,
                          message={'sender': 'job_logger', **row})

    def _bq_insert(self, row: dict):
        """
        Insert scores into BigQuery table

        :param row: The row to insert
        """
        table = f'{config.gcp_project}.{config.bq_dataset}.{self._workflow_name}'

        # Make a deep copy of the row so that we don't modify the original
        row = deepcopy(row)
        # Set create_ts to the bigquery timestamp format
        row['create_ts'] = utcnow_str(fmt=BIGQUERY_TIMESTAMP_FORMAT)
        # JSON fields in BigQuery must be inserted as JSON strings... makes sense
        data = row.get('data')
        if data:
            if isinstance(obj, torch.Tensor):
                # Convert tensor to a Python scalar or list
                data = obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
            # If data is not a dict, wrap it in a dict
            if not isinstance(data, dict):
                data = {'value': data}
            # Use the AutoJSONEncoder to serialize the data,
            # to allow for custom serialization
            data_json = json.dumps(data, cls=AutoJSONEncoder)
            row['data'] = data_json
        # Actually insert the row into BigQuery
        insert_to_biqquery(table, row)


class TorchtunewrapperLoggerAgent(BaseJobLoggerAgent):
    """
    An agent used to log fine-tuning scores on the filesystem and on bigquery
    """

    @property
    def _workflow_name(self) -> str:
        return 'torchtunewrapper'

    def log_step(self, gpu_rank: int,
                 step_num: int, step_len: int, step_loss: float, step_lr: float,
                 step_peak_memory_active: int, step_peak_memory_alloc: int, step_peak_memory_reserved: int,
                 step_time_elapsed_s: float,
                 epoch_num: int, epoch_len: int):
        """
        Log the fine-tuning step scores

        :param gpu_rank: The GPU rank (i.e. the GPU number)
        :param step_num: The step number
        :param step_len: The step length
        :param step_loss: The step loss
        :param step_lr: The step learning rate
        :param step_peak_memory_active: The step peak memory active
        :param step_peak_memory_alloc: The step peak memory alloc
        :param step_peak_memory_reserved: The step peak memory reserved
        :param step_time_elapsed_s: The step time elapsed in seconds
        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :return:
        """
        data = {
            'gpu_rank': gpu_rank,
            'step_loss': step_loss,
            'step_lr': step_lr,
            'step_num': step_num,
            'step_len': step_len,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'step_time_elapsed_s': step_time_elapsed_s,
            'step_peak_memory_active': step_peak_memory_active,
            'step_peak_memory_alloc': step_peak_memory_alloc,
            'step_peak_memory_reserved': step_peak_memory_reserved,
        }
        self._log_data(operation='step', data=data)

    def log_epoch(self, gpu_rank: int, epoch_num: int, epoch_len: int, epoch_time_elapsed_s: float):
        """
        Log the fine-tuning epochs

        :param gpu_rank: The GPU rank (i.e. the GPU number)
        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :param epoch_time_elapsed_s: The epoch time elapsed in seconds
        :return:
        """
        data = {
            'gpu_rank': gpu_rank,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'epoch_time_elapsed_s': epoch_time_elapsed_s,
        }
        self._log_data(operation='epoch', data=data)

    def log_weights(self, weight_files: list, other_files: list):
        """
        Log the fine-tuning results (e.g. weight files, other files)

        :param weight_files: The fine-tuning weight files
        :param other_files: Other fine-tuning files, such as weights configuration files
        :return:
        """
        results_bucket_name = get_results_bucket()
        data = {
            'base_url': f'https://storage.googleapis.com/'
                        f'{results_bucket_name}/{self._user_id}/{self._job_id}',
            'weight_files': weight_files,
            'other_files': other_files
        }
        self._log_data(operation='weights', data=data)
