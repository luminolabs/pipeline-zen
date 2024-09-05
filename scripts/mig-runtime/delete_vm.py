#!python

import logging
import argparse
import os

from google.cloud import compute_v1

from common.gcp import get_vm_name_from_metadata, get_zone_from_metadata
from utils import PROJECT_ID

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def delete_instance(project_id: str, zone: str, instance_name: str) -> None:
    """
    Delete a VM instance that's not part of a Managed Instance Group.

    Args:
        project_id (str): The Google Cloud project ID.
        zone (str): The zone of the VM instance.
        instance_name (str): The name of the VM instance to delete.
    """
    # Initialize the Compute Engine client
    instances_client = compute_v1.InstancesClient()

    # Create the delete request
    request = compute_v1.DeleteInstanceRequest(
        project=project_id,
        zone=zone,
        instance=instance_name
    )

    # Delete the VM instance
    print(f"Deleting VM {instance_name} in zone {zone}")
    operation = instances_client.delete(request=request)

    # Wait for the operation to complete
    while not operation.done():
        operation.result()

    if operation.error:
        print(f"Error deleting VM: {operation.error}")
    else:
        print(f"VM {instance_name} deleted successfully")


if __name__ == '__main__':
    # Determine if arguments are required based on the environment
    args_required = os.environ.get('PZ_ENV', 'local') == 'local'

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Delete a VM instance from a Managed Instance Group.')
    parser.add_argument('--vm_name', type=str, help='Name of the VM instance', required=args_required)
    parser.add_argument('--vm_zone', type=str, help='Zone of the VM instance', required=args_required)
    args = parser.parse_args()

    # Get the values from the metadata server if not provided as arguments
    vm_name = args.vm_name if args.vm_name else get_vm_name_from_metadata()
    vm_zone = args.vm_zone if args.vm_zone else get_zone_from_metadata()

    logger.info(f'Initiating deletion of VM {vm_name} in zone {vm_zone}')
    # Delete the VM from the MIG
    delete_instance(PROJECT_ID, vm_zone, vm_name)
    logger.info('VM deletion process completed')
