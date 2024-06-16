#!python

import argparse
import sys
import subprocess
import time
import uuid

from google.cloud import compute_v1

from delete_vm import delete_vm

# Configuration (adjust as needed)
PROJECT_ID = 'neat-airport-407301'
REGION = 'us-central1'
ZONE = f'{REGION}-a'
BASE_TEMPLATE_NAME = 'ubuntu-1xv100-pipeline-zen-jobs'
JOB_DIRECTORY = '/pipeline-zen-jobs'


def main(version: str, job_config_name: str, job_id: str, *args):
    """
    Run the model training workflow remotely on a compute instance.

    :param version: The version of the template to use
    :param job_config_name: The name of the job config file, without the `.py` extension
    :param job_id: The job_id to use for training; logs and other job results and artifacts will be named after this.
    :param args: Arguments to pass from the CLI to the job
    :return:
    """

    # Create auto-generated job id if one is not given
    job_id = job_id or (job_config_name + '-' + str(uuid.uuid4()))

    vm_name = f'{BASE_TEMPLATE_NAME}-{job_id}'.replace('_', '-')
    print(f'Creating VM from template: {vm_name}')

    # Create VM instance from template
    instance_client = compute_v1.InstancesClient()
    instance_insert_request = compute_v1.InsertInstanceRequest()
    instance_insert_request.project = PROJECT_ID
    instance_insert_request.zone = ZONE
    instance_insert_request.source_instance_template = \
        f'global/instanceTemplates/{BASE_TEMPLATE_NAME}-{version}'
    instance_insert_request.instance_resource.name = vm_name

    try:
        # Insert the compute instance using the template
        operation = instance_client.insert(instance_insert_request)
        # Wait for operation to complete
        operation.result()
        print(f'...VM created')
    except Exception as e:
        print(f'Failed to create VM: {e}')
        return

    # Wait for VM to start
    print('Starting VM...')
    while True:
        instance_resource = instance_client.get(project=PROJECT_ID, zone=ZONE, instance=vm_name)
        if instance_resource.status == 'RUNNING':
            print('...VM is running')
            # Wait for sshd to start
            print('...Wait for sshd to start (60s)')
            time.sleep(60)
            break
        time.sleep(5)
        print('...still waiting for VM to start')

    # Set up VM CLI command prefix, to be used in ssh commands below
    cmd_prefix = ['gcloud', 'compute', 'ssh', '--zone', ZONE, vm_name, '--command']

    # Execute job directly (change directory and run command)
    job_command = (f'cd {JOB_DIRECTORY} && source ./scripts/export-secrets.sh && '
                   f'PZ_ENV=dev ./scripts/run-celery-docker.sh {" ".join(args)}')
    try:
        # This will monitor and echo job output
        print(f'Running job: {job_id}')
        print('...this might take a while!')
        print('...note: the job will continue to run, even if we disconnect')
        print('...note: the job will stop the VM when done, even if we disconnect')
        print('!!! If you\'re asked for a password here, that\'s your Google password;')
        print('!!! the `gcloud` command is asking for your password, not this script;')
        print('!!! this script runs the `gcloud` command to start the job on the remote VM')
        time.sleep(5)  # pause for user to ack message
        subprocess.run([*cmd_prefix, job_command], check=True)
    except Exception as ex:
        # Delete VM if we couldn't start the job
        print('We failed to start the job or something else went wrong!!!')
        delete_vm(instance_client, vm_name)
        raise ex

    # The job itself will stop and delete the VM when done,
    # so if we're here, the VM is deleted.
    print('Done.')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the model training workflow remotely")
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help="The name of the job config file, without the `.py` extension")
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help="The job_id to use for training; "
                             "logs and other job results and artifacts will be named after this.")
    known_args, _ = parser.parse_known_args()
    job_config_name, job_id = known_args.job_config_name, known_args.job_id

    # Get version from VERSION file
    with open('VERSION', 'r') as f:
        version = f.read().strip().replace('.', '-')

    # Get all arguments
    all_args = sys.argv[1:]  # Skip the first element which is the script name
    # Run the workflow remotely
    main(version, job_config_name, job_id, *all_args)
