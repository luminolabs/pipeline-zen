#!python

import argparse
import time

from google.cloud import compute_v1

# Configuration (adjust as needed)
PROJECT_ID = 'neat-airport-407301'
ZONE = 'us-central1-a'
TEMPLATE_NAME = 'ubuntu-1xv100-pipeline-zen-jobs'


def main(job_id):
    vm_name = f'{TEMPLATE_NAME}-{job_id}'
    instance_client = compute_v1.InstancesClient()
    delete_vm(instance_client, vm_name)


def delete_vm(instance_client, vm_name: str) -> None:
    """
    Delete the VM

    :param instance_client: The VM instance client
    :param vm_name: Name of the VM to delete
    :return:
    """
    # First, allow logs to flush to GCP logging service
    print('Flushing logs to GCP...')
    time.sleep(10)

    print('Deleting VM...')
    operation = instance_client.delete(project=PROJECT_ID, zone=ZONE, instance=vm_name)
    operation.result()
    print('...VM deleted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', required=True, help='Used to locate the VM instance to delete')
    args = parser.parse_args()

    main(
        args.job_id.strip(),
    )
