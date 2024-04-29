from datetime import datetime
from logging import Logger
from typing import Optional

from google.cloud import bigquery

datetime_format = '%Y-%m-%d %H:%M:%S'
bq_table_train = 'neat-airport-407301.pipeline_zen.train'


class TrainScoresAgent:
    def __init__(self, job_id: str, logger: Logger):
        self.job_id = job_id
        self.logger = logger
        self.time_start = None
        self.time_end = None
        self.bq = bigquery.Client(project='neat-airport-407301')

    def mark_time_start(self):
        self.time_start = datetime.now()
        str_time = self.time_start.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f'Process started at: {str_time}')
        self.bq_insert(operation='mark_time_start', metric=str_time)

    def mark_time_end(self):
        self.time_end = datetime.now()
        str_time = self.time_end.strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f'Process ended at: {str_time}')
        self.bq_insert(operation='mark_time_end', metric=str_time)

    def log_batch(self, batch_num: int, batch_len: int, batch_loss: float, epoch_num: int, epoch_len: int):
        self.logger.info(f'Batch #{batch_num}/{batch_len}, Loss: {batch_loss:.4f}')
        self.bq_insert(operation='log_batch', **{
            'batch_num': batch_num,
            'batch_len': batch_len,
            'batch_loss': batch_loss,
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
        })

    def log_epoch(self, epoch_num: int, epoch_len: int, epoch_loss: float):
        self.logger.info(f'Epoch #{epoch_num}/{epoch_len}, Loss: {epoch_loss:.4f}')
        self.bq_insert(operation='log_epoch', **{
            'epoch_num': epoch_num,
            'epoch_len': epoch_len,
            'epoch_loss': epoch_loss
        })

    def log_time_elapsed(self):
        time_delta_m = (self.time_start - self.time_end).seconds / 60
        self.logger.info(f'Elapsed time: {time_delta_m}')
        self.bq_insert(operation='log_time_elapsed', metric=time_delta_m)

    def bq_insert(self, operation: str, metric: Optional[str] = None, **kwargs):
        row = {
            **{
                'job_id': self.job_id,
                'create_ts': str(datetime.now()),
                'operation': operation,
                'metric': metric,
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
