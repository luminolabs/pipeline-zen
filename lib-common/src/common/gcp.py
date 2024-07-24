import requests

METADATA_ZONE_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/zone'
METADATA_NAME_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/name'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
LOCAL_ENV = 'local'
LOCAL_CLUSTER = LOCAL_ENV
STORAGE_BUCKET_PREFIX = 'lum-pipeline-zen-jobs'


# Function to get the VM name using the metadata server
def get_vm_name_from_metadata():
    print('Fetching the VM name from the metadata server...')
    response = requests.get(METADATA_NAME_URL, headers=METADATA_HEADERS)
    vm_name = response.text
    print(f'VM name obtained: {vm_name}')
    return vm_name


# Function to get the zone of the VM from the metadata server
def get_zone_from_metadata():
    print('Fetching the zone from the metadata server...')
    response = requests.get(METADATA_ZONE_URL, headers=METADATA_HEADERS)
    zone = response.text.split('/')[-1]
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
