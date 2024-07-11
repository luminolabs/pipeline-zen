#!/usr/bin/env python3

import logging
import argparse
import os

from google.cloud import compute_v1

from utils import PROJECT_ID, get_mig_name_from_vm_name, get_vm_name_from_metadata, get_zone_from_metadata, \
    get_region_from_zone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def delete_instance_from_mig(project_id: str, vm_zone: str, mig_name: str, vm_name: str) -> None:
    """
    Delete an instance from a regional Managed Instance Group (MIG).

    Args:
        project_id (str): The Google Cloud project ID.
        vm_zone (str): The zone of the VM instance.
        mig_name (str): The name of the Managed Instance Group.
        vm_name (str): The name of the VM instance to delete.
    """
    # Get the region from the zone
    vm_region = get_region_from_zone(vm_zone)

    # Initialize the regional MIG VM service
    instance_group_managers_client = compute_v1.RegionInstanceGroupManagersClient()

    # Create the VM delete request
    request = compute_v1.DeleteInstancesRegionInstanceGroupManagerRequest(
        instance_group_manager=mig_name,
        region_instance_group_managers_delete_instances_request_resource=
        compute_v1.RegionInstanceGroupManagersDeleteInstancesRequest(
            instances=[f'zones/{vm_zone}/instances/{vm_name}']
        ),
        project=project_id,
        region=vm_region,
    )

    # Delete VM from MIG
    logger.info(f"Deleting VM {vm_name} from MIG {mig_name} in zone {vm_zone}")
    operation = instance_group_managers_client.delete_instances(request=request)
    # Wait for the operation to complete
    operation.result()
    logger.info(f"VM {vm_name} deleted successfully")


if __name__ == '__main__':
    # Determine if arguments are required based on the environment
    args_required = os.environ.get('PZ_ENV', 'local') == 'local'

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Delete a VM instance from a Managed Instance Group.')
    parser.add_argument('--vm_name', type=str, help='Name of the VM instance', required=args_required)
    parser.add_argument('--vm_zone', type=str, help='Zone of the VM instance', required=args_required)
    parser.add_argument('--mig_name', type=str, help='MIG of the VM instance', required=args_required)
    args = parser.parse_args()

    # Get the values from the metadata server if not provided as arguments
    vm_name = args.vm_name if args.vm_name else get_vm_name_from_metadata()
    vm_zone = args.vm_zone if args.vm_zone else get_zone_from_metadata()
    mig_name = args.mig_name if args.mig_name else get_mig_name_from_vm_name(vm_name)

    logger.info(f'Initiating deletion of VM {vm_name} from MIG {mig_name} in zone {vm_zone}')
    # Delete the VM from the MIG
    delete_instance_from_mig(PROJECT_ID, vm_zone, mig_name, vm_name)
    logger.info('VM deletion process completed')