import argparse
import subprocess
from time import sleep

import requests
from google.cloud import compute_v1
from google.cloud.compute_v1 import InstancesClient

PROJECT_ID = 'neat-airport-407301'


def get_vm_name_from_uname():
    """
    Get the VM name from the `uname -n` command
    :return: The VM name
    """
    return subprocess.check_output('uname -n', shell=True).decode().strip()


def get_zone():
    """
    Get the zone of the VM from the metadata server
    :return: The zone of the VM
    """
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"}
    )
    zone = response.text.strip()
    return zone.split('/')[-1]


def delete_vm(instance_client: InstancesClient, vm_name: str, vm_zone: str) -> None:
    """
    Delete the VM

    :param instance_client: The instance client to use
    :param vm_name: Name of the VM to delete
    :param vm_zone: Zone of the VM
    :return:
    """
    # Allow logs to flush to GCP logging service
    print('Flushing logs to GCP...')
    sleep(10)

    print(f"Deleting VM: {vm_name} in zone: {vm_zone} for project: {PROJECT_ID}...")
    operation = instance_client.delete(project=PROJECT_ID, zone=vm_zone, instance=vm_name)
    operation.result()
    print('...VM deleted')


def main():
    parser = argparse.ArgumentParser(description="Get VM name and zone.")
    parser.add_argument('--vm_name', type=str, help='The VM name to delete')
    parser.add_argument('--vm_zone', type=str, help='The zone of the VM to delete')
    args = parser.parse_args()

    # Get the VM name and zone
    if args.vm_name:
        vm_name = args.vm_name
    else:
        vm_name = get_vm_name_from_uname()
    if args.vm_name:
        vm_zone = args.vm_zone
    else:
        vm_zone = get_zone()

    # Delete the VM
    instance_client = compute_v1.InstancesClient()
    delete_vm(instance_client, vm_name, vm_zone)

if __name__ == "__main__":
    main()
