from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import Optional, Union
import json

from google.cloud import bigquery

from common.agents.system_metrics import SystemSpecs

datetime_format = '%Y-%m-%d %H:%M:%S'
bq_table_train = 'neat-airport-407301.pipeline_zen.train'
bq_table_evaluate = 'neat-airport-407301.pipeline_zen.evaluate'


class BaseScoresAgent(ABC):
    """
    Base class for capturing model related scores
    """
    def __init__(self, job_id: str, logger: Logger):
        """
        :param job_id: The id of the job
        :param logger: The logger to use
        :return:
        """
        self.job_id = job_id
        self.logger = logger
        self.time_start = None
        self.time_end = None

        # Configure bigquery
        self.bq_table = self._get_bq_table()
        bq_project = self.bq_table.split('.')[0]
        self.bq = bigquery.Client(bq_project)
        self.bq_table_defaults = self._get_bq_table_defaults()
        # Get a copy of the system specs
        self.system_specs = SystemSpecs(logger)

    def mark_time_start(self):
        """
        Mark, and log the start time of the training job
        """
        self.time_start = datetime.now()
        str_time = self.time_start.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f'Process started at: {str_time}')
        self.bq_insert(operation='mark_time_start', result=str_time)

    def mark_time_end(self):
        """
        Mark, and log the end time of the training job
        """
        self.time_end = datetime.now()
        str_time = self.time_end.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f'Process ended at: {str_time}')
        self.bq_insert(operation='mark_time_end', result=str_time)

    def log_time_elapsed(self):
        """
        Log the training time length in minutes
        :return:
        """
        time_delta_m = f'{(self.time_end - self.time_start).total_seconds() / 60:.2f} minutes'
        self.logger.info(f'Elapsed time: {time_delta_m}')
        self.bq_insert(operation='log_time_elapsed', result=time_delta_m)

    def log_system_specs(self):
        """
        Log system specs
        :return:
        """
        system_specs = self.system_specs.get_specs()
        self.logger.info(f'System specs: {system_specs}')
        self.bq_insert(operation='log_system_specs', result=system_specs)

    def bq_insert(self, operation: str, result: Optional[Union[dict, str]] = None, **kwargs):
        """
        Insert scores into BigQuery table

        :param operation: The operation name of the score to be inserted
        :param result: The value of the score to be inserted (optional)
        :param kwargs: Dict with scores to be inserted (see `row` contents below)
        :return:
        """

        # Normalize result
        result_json = None
        if result:
            if not isinstance(result, dict):
                result = {'value': result}
            result_json = json.dumps(result)

        # Construct row
        row = {
            # Create a new dict from the dicts below;
            # the new dict represents the target table structure
            **{'job_id': self.job_id,
                'create_ts': str(datetime.now()),
                'operation': operation,
                'result': result_json},
            **self.bq_table_defaults,
            **kwargs}

        # Handle errors
        errors = self.bq.insert_rows_json(self.bq_table, [row])
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
        self.bq_insert(operation='log_batch', **{
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
        self.bq_insert(operation='log_epoch', **{
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'epoch_loss': epoch_loss
        })


class EvaluateScoresAgent(BaseScoresAgent):
    """
    An agent used to log evaluate scores on the filesystem and on bigquery
    """

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
        self.bq_insert(operation='log_scores', **{
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
