from abc import ABC
from datetime import datetime
from logging import Logger
from typing import Optional

from google.cloud import bigquery

datetime_format = '%Y-%m-%d %H:%M:%S'
bq_table_train = 'neat-airport-407301.pipeline_zen.train'


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
        project = bq_table_train.split('.')[0]
        self.bq = bigquery.Client(project)

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
        time_delta_m = (self.time_start - self.time_end).seconds / 60
        self.logger.info(f'Elapsed time: {time_delta_m}')
        self.bq_insert(operation='log_time_elapsed', result=time_delta_m)

    def bq_insert(self, operation: str, result: Optional[str] = None, **kwargs):
        """
        Insert scores into BigQuery table

        :param operation: The operation name of the score to be inserted
        :param result: The value of the score to be inserted (optional)
        :param kwargs: Dict with scores to be inserted (see `row` contents below)
        :return:
        """
        row = {
            **{
                'job_id': self.job_id,
                'create_ts': str(datetime.now()),
                'operation': operation,
                'result': result,
                'batch_num': None,
                'batch_len': None,
                'batch_loss': None,
                'epoch_num': None,
                'epoch_len': None,
                'epoch_loss': None
            },
            **kwargs}
        errors = self.bq.insert_rows_json(bq_table_train, [row])
        if errors:
            raise SystemError('Encountered errors while inserting rows: {}'.format(errors))


class TrainScoresAgent(BaseScoresAgent):
    """
    An agent used to log training scores on the filesystem and on bigquery
    """

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
    pass
