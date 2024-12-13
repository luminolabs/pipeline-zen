import glob
import json
import os
from typing import Optional

import requests
from filelock import FileLock
from google.api_core.exceptions import TooManyRequests
from google.cloud import pubsub_v1, storage, bigquery

from common.config_manager import config
from common.utils import setup_logger, is_local_env, get_work_dir

# GCP Metadata URLs and headers; only used in GCP VMs
METADATA_ZONE_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/zone'
METADATA_NAME_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/name'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
# BigQuery timestamp format
BIGQUERY_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# Initialize the BigQuery and Pub/Sub clients to be used across this module
bigquery_client = bigquery.Client(config.gcp_project)
pubsub_publisher_client = pubsub_v1.PublisherClient()
storage_client = storage.Client(project=config.gcp_project)


def insert_to_biqquery(table: str, row: dict):
    """
    Insert rows into a BigQuery table

    :param table: The table to insert into
    :param row: The row to insert
    """
    if not config.send_to_bq:
        return

    errors = bigquery_client.insert_rows_json(table, [row])
    if errors:
        raise SystemError('Encountered errors while inserting rows: {}'.format(errors))


def publish_to_pubsub(job_id: str, user_id: str, topic_name: str, message: dict):
    """
    Send a message to a Pub/Sub topic.

    :param job_id: The job id
    :param user_id: The user id
    :param topic_name: The name of the topic
    :param message: The message to send
    """
    if not config.send_to_pubsub:
        return

    logger = setup_logger(f'send_message_to_pubsub', job_id, user_id)
    message.update({'job_id': job_id, 'user_id': user_id})
    topic_path = pubsub_publisher_client.topic_path(config.gcp_project, topic_name)
    message_str = json.dumps(message)
    logger.info(f'Sending message to Pub/Sub topic {topic_name}: {message_str}')
    pubsub_publisher_client.publish(topic_path, message_str.encode("utf-8"))


def get_results_bucket() -> str:
    """
    Get the results bucket name.

    We maintain buckets for the `us`, `asia`, and `europe` multi-regions.
    We have a regional bucket for `me-west1`, because Middle East doesn't
    have multi-region storage infrastructure on GCP.

    ex.
    - 'pipeline-zen-jobs-8xa100-40gb-us-central1-asj3' -> 'pipeline-zen-jobs-us'
    - 'pipeline-zen-jobs-8xa100-40gb-me-west1-ki3d' -> 'pipeline-zen-jobs-me-west1'

    :return: The results bucket name
    """
    bucket_prefix = f'lum-{config.env_name}-pipeline-zen-jobs'

    # If set, use results_bucket_suffix instead of computing it
    if config.results_bucket_suffix:
        return f'{bucket_prefix}-{config.results_bucket_suffix}'
    elif is_local_env():
        return f'{bucket_prefix}-us'

    # Get zone, region, and multi-region from metadata
    zone = get_zone_from_metadata()
    region = get_region_from_zone(zone)
    multi_region = get_multi_region_from_zone(zone)

    # Middle East doesn't have a multi-region storage configuration on GCP,
    # so we maintain a regional bucket for `me-west1`.
    if multi_region == 'me':
        return f'{bucket_prefix}-{region}'  # regional bucket; ie. 'pipeline-zen-jobs-me-west1'
    return f'{bucket_prefix}-{multi_region}'  # multi-region bucket; ie. 'pipeline-zen-jobs-us'


def upload_directory(local_path: str, bucket: Optional[str] = None, gcs_path: Optional[str] = None):
    """
    Upload a local directory to Google Cloud Storage.

    :param local_path: Local path to upload
    :param bucket: Bucket to upload to
    :param gcs_path: GCS folder to upload to
    :return:
    """
    if not config.send_to_gcs:
        return

    # Set the bucket to the default results bucket if not provided
    bucket = bucket or get_results_bucket()
    # Instantiate the bucket object
    bucket = storage_client.get_bucket(bucket)

    # If the GCS path is not set, use the last two directories of the local path
    # i.e. go from ./.results/user_id/job_id to user_id/job_id
    gcs_path = gcs_path or '/'.join(local_path.split('/')[-2:])

    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_directory(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def upload_file(local_file: str, bucket: Optional[str] = None, gcs_path: Optional[str] = None):
    """
    Upload a local file to Google Cloud Storage.

    :param local_file: Local path to upload
    :param bucket: Bucket to upload to
    :param gcs_path: GCS folder to upload to
    :return:
    """
    if not config.send_to_gcs:
        return

    # Set the bucket to the default results bucket if not provided
    bucket = bucket or get_results_bucket()
    # Instantiate the bucket object
    bucket = storage_client.get_bucket(bucket)

    # If the GCS path is not set, use the last two directories of the local path
    # i.e. go from ./.results/user_id/job_id/file to user_id/job_id
    gcs_path = gcs_path or '/'.join(local_file.split('/')[-3:-1])

    assert os.path.isfile(local_file)
    remote_path = os.path.join(gcs_path, os.path.basename(local_file))
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_file)


def upload_jobs_meta(job_id: str, user_id: str):
    """
    Upload the job metadata to Google Cloud Storage.

    :param job_id: The job id
    :param user_id: The user id
    """
    logger = setup_logger('upload_jobs_meta', job_id, user_id)
    logger.info(f'Uploading job metadata to GCS; job_id: {job_id}, user_id: {user_id}')
    # Get the local path to the job metadata
    path = os.path.join(get_work_dir(job_id, user_id), config.job_meta_file)
    if not os.path.exists(path):
        logger.warning(f'Job metadata file does not exist (yet): {path}')
        return
    bucket = get_results_bucket()
    with FileLock(path + '.lock', thread_local=False):
        # Upload the job metadata to GCS
        try:
            upload_file(path, bucket)
        except TooManyRequests as e:
            logger.warning(f'Got too many requests error while uploading job metadata - skipping: {e}')


def make_object_public(bucket_name, object_name):
    """
    Makes a specific object in a Google Cloud Storage bucket public and returns its public URL.

    Args:
        bucket_name (str): The name of the GCS bucket.
        object_name (str): The name of the object (file) in the GCS bucket.

    Returns:
        str: The public URL of the object.
    """
    if not config.send_to_gcs:
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.make_public()
    public_url = blob.public_url
    return public_url


def download_object(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """
    Downloads a file from a Google Cloud Storage bucket.

    Args:
        bucket_name: The name of the bucket
        source_blob_name: The name of the source blob
        destination_file_name: The name of the destination file
    """
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Get the blob
    blob = bucket.blob(source_blob_name)
    # Download the blob to a local file
    blob.download_to_filename(destination_file_name)


#########################
### VM Metadata Utils ###
#########################

# NOTE: These only work when running on GCP VMs

def get_vm_name_from_metadata():
    """
    Get the VM name from the metadata server
    """
    if not config.is_gcp:
        return None
    print('Fetching the VM name from the metadata server...')
    if config.vm_name:
        return config.vm_name
    response = requests.get(METADATA_NAME_URL, headers=METADATA_HEADERS)
    vm_name = response.text
    config.set('vm_name', vm_name)
    print(f'VM name obtained: {vm_name}')
    return vm_name


def get_zone_from_metadata():
    """
    Get the zone of the VM from the metadata server
    """
    if not config.is_gcp:
        return None
    print('Fetching the zone from the metadata server...')
    if config.zone:
        return config.zone
    response = requests.get(METADATA_ZONE_URL, headers=METADATA_HEADERS)
    zone = response.text.split('/')[-1]
    config.set('zone', zone)
    print(f'Zone obtained: {zone}')
    return zone


def get_mig_name_from_vm_name(vm_name: str) -> str:
    """
    Get the MIG name from the VM name

    ex. 'pipeline-zen-jobs-8xa100-40gb-us-central1-asj3' -> 'pipeline-zen-jobs-8xa100-40gb-us-central1'

    :param vm_name: The name of the VM
    :return: The name of the MIG
    """
    return '-'.join(vm_name.split('-')[:-2]) + '-mig'


def get_region_from_zone(zone: str) -> str:
    """
    Get the region from the zone

    ex. 'us-central1-a' -> 'us-central1'

    :param zone: The zone
    :return: The region
    """
    return '-'.join(zone.split('-')[:-1])


def get_multi_region_from_zone(zone: str) -> str:
    """
    Get the multi-region from the VM name.
    The multi-region is the wider region that the VM belongs to, e.g., 'us' or 'asia' or 'europe'.

    ex. 'us-central1-a' -> 'us'

    :param zone: The zone
    :return: The multi-region
    """
    return zone.split('-')[0]


def get_region_from_vm_name(vm_name: str) -> str:
    """
    Get the region from the VM name

    ex. 'pipeline-zen-jobs-8xa100-40gb-us-central1-asj3' -> 'us-central1'

    :param vm_name: The name of the VM
    :return: The region
    """
    return '-'.join(vm_name.split('-')[-4:-2])
