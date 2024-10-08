import json
from typing import Optional

import requests
from google.cloud import pubsub_v1, storage

from common.config_manager import config
from common.utils import utcnow_str, setup_logger

METADATA_ZONE_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/zone'
METADATA_NAME_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/name'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
LOCAL_ENV = 'local'
LOCAL_CLUSTER = LOCAL_ENV
STORAGE_BUCKET_PREFIX = 'lum-pipeline-zen-jobs'


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
    send_message_to_pubsub(job_id, user_id, config.heartbeat_topic, msg)


def send_message_to_pubsub(job_id: str, user_id: str, topic_name: str, message: dict):
    """
    Send a message to a Pub/Sub topic.

    :param job_id: The job id
    :param user_id: The user id
    :param topic_name: The name of the topic
    :param message: The message to send
    """
    logger = setup_logger(f'send_message_to_pubsub', job_id, user_id)
    message.update({'job_id': job_id, 'user_id': user_id})
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(config.gcp_project, topic_name)
    message_str = json.dumps(message)
    logger.info(f'Sending message to Pub/Sub topic {topic_name}: {message_str}')
    publisher.publish(topic_path, message_str.encode("utf-8"))


def get_vm_name_from_metadata():
    """
    Get the VM name from the metadata server
    """
    if config.env_name == LOCAL_ENV:
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
    if config.env_name == LOCAL_ENV:
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
    return '-'.join(vm_name.split('-')[:-1])


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
    return '-'.join(vm_name.split('-')[-3:-1])


def get_results_bucket_name(env_name: str) -> str:
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
    # If running locally, use the local dev bucket
    if env_name == LOCAL_ENV:
        return f'{STORAGE_BUCKET_PREFIX}-{LOCAL_CLUSTER}'  # ie. 'pipeline-zen-jobs-local'

    # Get zone, region, and multi-region from metadata
    zone = get_zone_from_metadata()
    region = get_region_from_zone(zone)
    multi_region = get_multi_region_from_zone(zone)

    # Middle East doesn't have a multi-region storage configuration on GCP,
    # so we maintain a regional bucket for `me-west1`.
    if multi_region == 'me':
        return f'{STORAGE_BUCKET_PREFIX}-{region}'  # regional bucket; ie. 'pipeline-zen-jobs-me-west1'
    return f'{STORAGE_BUCKET_PREFIX}-{multi_region}'  # multi-region bucket; ie. 'pipeline-zen-jobs-us'


def make_gcs_object_public(bucket_name, object_name):
    """
    Makes a specific object in a Google Cloud Storage bucket public and returns its public URL.

    Args:
        bucket_name (str): The name of the GCS bucket.
        object_name (str): The name of the object (file) in the GCS bucket.

    Returns:
        str: The public URL of the object.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.make_public()
    public_url = blob.public_url
    return public_url
