from abc import ABC, abstractmethod
from copy import deepcopy
from logging import Logger
from typing import Optional, Union
import json

from google.cloud import bigquery

from common.agents.system_metrics import SystemSpecs
from common.config_manager import config
from common.gcp import send_message_to_pubsub
from common.utils import AutoJSONEncoder, utcnow, utcnow_str, job_meta_context

bq_table_train = f'{config.gcp_project}.{config.bq_dataset}.train'
bq_table_evaluate = f'{config.gcp_project}.{config.bq_dataset}.evaluate'
bq_table_torchtunewrapper = f'{config.gcp_project}.{config.bq_dataset}.torchtunewrapper'


class BaseScoresAgent(ABC):
    """
    Base class for capturing model related scores
    """
    def __init__(self, job_id: str, user_id: str, logger: Logger):
        """
        :param job_id: The id of the job
        :param user_id: The id of the user
        :param logger: The logger to use
        :return:
        """
        self.job_id = job_id
        self.user_id = user_id
        self.logger = logger
        self.time_start = None
        self.time_end = None

        # Configure bigquery
        self.bq_table = self._get_bq_table()
        self.bq = bigquery.Client(config.gcp_project) \
            if config.send_to_bq else None
        self.bq_table_defaults = self._get_bq_table_defaults()
        # Get a copy of the system specs
        self.system_specs = SystemSpecs(logger)

    @property
    @abstractmethod
    def workflow_name(self) -> str:
        pass

    def mark_time_start(self):
        """
        Mark, and log the start time of the training job
        """
        self.time_start = utcnow()
        self.logger.info(f'Process started at: {utcnow_str()}')
        self.log_data(operation='mark_time_start')

    def mark_time_end(self):
        """
        Mark, and log the end time of the training job
        """
        self.time_end = utcnow()
        self.logger.info(f'Process ended at: {utcnow_str()}')
        self.log_data(operation='mark_time_end')

    def log_time_elapsed(self):
        """
        Log the training time length in minutes
        :return:
        """
        time_delta_m = f'{(self.time_end - self.time_start).total_seconds() / 60:.2f} minutes'
        self.logger.info(f'Elapsed time: {time_delta_m}')
        self.log_data(operation='log_time_elapsed', result=time_delta_m)

    def log_system_specs(self):
        """
        Log system specs
        :return:
        """
        system_specs = self.system_specs.get_specs()
        self.logger.info(f'System specs: {system_specs}')
        self.log_data(operation='log_system_specs', result=system_specs)

    def log_job_config(self, job_config: dict):
        """
        Log job configuration

        :param job_config: The job configuration
        :return:
        """
        self.logger.info(f'Training job type: `{job_config["category"]}` - `{job_config["type"]}`')
        self.logger.info(f'Job configuration: {job_config}')
        self.log_data(operation='log_job_config', result=job_config)

    def log_data(self, operation: str, result: Optional[Union[dict, str]] = None, **kwargs):
        """
        Log data

        :param operation: The operation name of the score to be inserted
        :param result: The value of the score to be inserted (optional)
        :param kwargs: Dict with scores to be inserted (see `row` contents below)
        :return:
        """

        # Construct row
        row = {
            # Create a new dict from the dicts below;
            # the new dict represents the target table structure
            **{'job_id': self.job_id,
               'create_ts': utcnow_str(),
               'operation': operation,
               'result': result},
            **self.bq_table_defaults,
            **kwargs}

        # JSON the result if it's a dict
        row = deepcopy(row)
        result = row.get('result')
        if result:
            if not isinstance(result, dict):
                result = {'value': result}
            result_json = json.dumps(result, cls=AutoJSONEncoder)
            row['result'] = result_json

        self.bq_insert(row)
        self.pubsub_send(row)
        self.file_write(row)

    def file_write(self, row: dict):
        """
        Write scores to a file

        :param row: The row to write
        """
        with job_meta_context(self.job_id, self.user_id) as job_meta:
            job_meta.setdefault('scores', []).append(row)

    def pubsub_send(self, row: dict):
        """
        Send scores to Pub/Sub

        :param row: The row to send
        """
        send_message_to_pubsub(self.job_id, self.user_id,
                               topic_name=config.jobs_meta_topic,
                               message={'action': 'job_progress',
                                        'workflow': self.workflow_name,
                                        **row})

    def bq_insert(self, row: dict):
        """
        Insert scores into BigQuery table

        :param row: The row to insert
        """
        # If sending scores to BigQuery is disabled
        if not config.send_to_bq:
            return

        errors = self.bq.insert_rows_json(self.bq_table, [row])
        # Handle errors
        if errors:
            raise SystemError('Encountered errors while inserting rows: {}'.format(errors))

    @abstractmethod
    def _get_bq_table(self) -> str:
        pass

    @abstractmethod
    def _get_bq_table_defaults(self) -> dict:
        pass


class TrainScoresAgent(BaseScoresAgent):
    """
    An agent used to log training scores on the filesystem and on bigquery
    """

    @property
    def workflow_name(self) -> str:
        return 'train'

    def _get_bq_table(self) -> str:
        return bq_table_train

    def _get_bq_table_defaults(self) -> dict:
        return {
            'batch_num': None,
            'batch_len': None,
            'batch_loss': None,
            'epoch_num': None,
            'epoch_len': None,
            'epoch_loss': None
        }

    def log_batch(self, batch_num: int, batch_len: int, batch_loss: float, epoch_num: int, epoch_len: int):
        """
        Log the training batch scores

        :param batch_num: The batch number
        :param batch_len: The batch length
        :param batch_loss: The training batch loss
        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :return:
        """
        self.logger.info(f'Batch #{batch_num}/{batch_len}, Loss: {batch_loss:.4f}')
        self.log_data(operation='log_batch', **{
            'batch_num': batch_num,
            'batch_len': batch_len,
            'batch_loss': batch_loss,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
        })

    def log_epoch(self, epoch_num: int, epoch_len: int, epoch_loss: float):
        """
        Log the training epoch scores

        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :param epoch_loss: The training epoch loss
        :return:
        """
        self.logger.info(f'Epoch #{epoch_num}/{epoch_len}, Loss: {epoch_loss:.4f}')
        self.log_data(operation='log_epoch', **{
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'epoch_loss': epoch_loss
        })


class EvaluateScoresAgent(BaseScoresAgent):
    """
    An agent used to log evaluate scores on the filesystem and on bigquery
    """

    @property
    def workflow_name(self) -> str:
        return 'evaluate'

    def _get_bq_table(self) -> str:
        return bq_table_evaluate

    def _get_bq_table_defaults(self) -> dict:
        return {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
        }

    def log_scores(
            self, accuracy: float, precision: float, recall: float, f1: float,
            stopped_at: int, num_batches: int):
        self.logger.info(
            f'Stopped at batch: {stopped_at}/{num_batches}\n'
            f'Accuracy: {accuracy:.4f}, \n' +
            f'Precision: {precision:.4f}, \n' +
            f'Recall: {recall:.4f}, \n' +
            f'F1: {f1:.4f}')
        self.log_data(operation='log_scores', **{
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })


class TorchtunewrapperScoresAgent(BaseScoresAgent):
    """
    An agent used to log fine-tuning scores on the filesystem and on bigquery
    """

    @property
    def workflow_name(self) -> str:
        return 'torchtunewrapper'

    def _get_bq_table(self) -> str:
        return bq_table_torchtunewrapper

    def _get_bq_table_defaults(self) -> dict:
        return {
            'gpu_rank': None,
            'step_num': None,
            'step_len': None,
            'step_loss': None,
            'step_lr': None,
            'step_tokens_per_second': None,
            'step_tokens': None,
            'step_peak_memory_active': None,
            'step_peak_memory_alloc': None,
            'step_peak_memory_reserved': None,
            'epoch_num': None,
            'epoch_len': None,
        }

    def log_step(self, gpu_rank: int,
                 step_num: int, step_len: int, step_loss: float, step_lr: float,
                 step_tokens_per_second: float, step_tokens: int,
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
        :param step_tokens_per_second: The step tokens per second per GPU
        :param step_tokens: The step tokens
        :param step_peak_memory_active: The step peak memory active
        :param step_peak_memory_alloc: The step peak memory alloc
        :param step_peak_memory_reserved: The step peak memory reserved
        :param step_time_elapsed_s: The step time elapsed in seconds
        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :return:
        """
        self.logger.info(f'GPU #{gpu_rank}, '
                         f'step #{step_num}/{step_len}, Loss: {step_loss:.4f}, '
                         f'LR: {step_lr:.4f}, '
                         f'Tokens/s/GPU: {step_tokens_per_second:.4f}, Tokens: {step_tokens}, '
                         f'Peak memory active: {step_peak_memory_active}, '
                         f'Peak memory alloc: {step_peak_memory_alloc}, '
                         f'Peak memory reserved: {step_peak_memory_reserved}, '
                         f'Time elapsed (seconds): {step_time_elapsed_s}, '
                         f'Epoch #{epoch_num}/{epoch_len}')
        self.log_data(operation='log_step', **{
            'gpu_rank': gpu_rank,
            'step_num': step_num,
            'step_len': step_len,
            'step_loss': step_loss,
            'step_lr': step_lr,
            'step_tokens_per_second': step_tokens_per_second,
            'step_tokens': step_tokens,
            'step_peak_memory_active': step_peak_memory_active,
            'step_peak_memory_alloc': step_peak_memory_alloc,
            'step_peak_memory_reserved': step_peak_memory_reserved,
            'step_time_elapsed_s': step_time_elapsed_s,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
        })

    def log_epoch(self, gpu_rank: int, epoch_num: int, epoch_len: int, epoch_time_elapsed_s: float):
        """
        Log the fine-tuning epochs

        :param gpu_rank: The GPU rank (i.e. the GPU number)
        :param epoch_num: The epoch number
        :param epoch_len: The epoch length
        :param epoch_time_elapsed_s: The epoch time elapsed in seconds
        :return:
        """
        self.logger.info(f'GPU #{gpu_rank}, '
                         f'Epoch #{epoch_num}/{epoch_len}')
        self.log_data(operation='log_epoch', **{
            'gpu_rank': gpu_rank,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'epoch_time_elapsed_s': epoch_time_elapsed_s,
        })
