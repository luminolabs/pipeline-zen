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
    print("Executing `uname -n` to get the VM name...")
    vm_name = subprocess.check_output('uname -n', shell=True).decode().strip()
    print(f"VM name obtained: {vm_name}")
    return vm_name


def resize_mig(mig_name):
    client = compute_v1.RegionInstanceGroupManagersClient()

    print(f"Fetching the current size of the MIG: {mig_name}...")
    mig = client.get(project=PROJECT_ID, region='us-central1', instance_group_manager=mig_name)
    current_size = mig.target_size
    print(f"Current size of MIG {mig_name}: {current_size}")

    new_size = max(current_size - 1, 0)
    print(f"Resizing MIG {mig_name} to new size: {new_size}...")

    operation = client.resize(project=PROJECT_ID, region='us-central1', instance_group_manager=mig_name, size=new_size)

    print("Waiting for resize operation to complete...")
    operation.result()
    print(f"MIG {mig_name} resized from {current_size} to {new_size}")


def get_zone():
    """
    Get the zone of the VM from the metadata server
    :return: The zone of the VM
    """
    print("Fetching the zone from the metadata server...")
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"}
    )
    zone = response.text.strip()
    print(f"Zone obtained: {zone}")
    return zone.split('/')[-1]


def delete_vm(instance_client: InstancesClient, vm_name: str, vm_zone: str) -> None:
    """
    Delete the VM

    :param instance_client: The instance client to use
    :param vm_name: Name of the VM to delete
    :param vm_zone: Zone of the VM
    :return:
    """
    print('Flushing logs to GCP...')
    sleep(10)

    print(f"Deleting VM: {vm_name} in zone: {vm_zone} for project: {PROJECT_ID}...")
    operation = instance_client.delete(project=PROJECT_ID, zone=vm_zone, instance=vm_name)

    print("Waiting for delete operation to complete...")
    operation.result()
    print(f"VM {vm_name} deleted successfully.")


def main():
    vm_name = get_vm_name_from_uname()
    vm_zone = get_zone()

    print(f"VM name - {vm_name}, VM zone - {vm_zone}")

    instance_client = compute_v1.InstancesClient()
    delete_vm(instance_client, vm_name, vm_zone)

    mig_name = '-'.join(vm_name.split('-')[:-1])
    print(f"Resizing MIG: {mig_name}")
    resize_mig(mig_name)


if __name__ == "__main__":
    main()
