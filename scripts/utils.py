import requests
from dotenv import load_dotenv

PROJECT_ID = 'neat-airport-407301'
METADATA_ZONE_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/zone'
METADATA_NAME_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/name'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
LOCAL_ENV = 'local'
LOCAL_CLUSTER = LOCAL_ENV

# Load environment variables from the .env file
load_dotenv()


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
