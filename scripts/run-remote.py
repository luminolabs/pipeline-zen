import argparse
import os
import subprocess
import time

from google.cloud import compute_v1

# TODO: Make configuration configurable
# see: https://linear.app/luminoai/issue/LUM-178/make-machine-type-configurable-when-deploying-job

# Configuration (adjust as needed)
PROJECT_ID = 'neat-airport-407301'
ZONE = 'us-central1-a'
SERVICE_ACCOUNT_EMAIL = f'pipeline-zen-jobs-dev@{PROJECT_ID}.iam.gserviceaccount.com'
TEMPLATE_NAME = 'ubuntu-1xv100-pipeline-zen-jobs'
IMAGE_NAME = 'ubuntu-pipeline-zen-jobs'
MACHINE_TYPE = 'n1-highcpu-8'
GPU = 'nvidia-tesla-v100'
JOB_DIRECTORY = '/pipeline-zen-jobs'
JOB_COMPLETION_FILE = os.path.join(JOB_DIRECTORY, '.results', '.finished')


def main(job_config_name, job_id, batch_size, num_epochs, num_batches):

    # Network Interface Configuration
    network_interface = compute_v1.NetworkInterface()
    network_interface.name = "global/networks/default"  # Use the default network
    network_interface.access_configs = [
        compute_v1.AccessConfig(
            name="External NAT",
            type_=compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name,
            network_tier="PREMIUM",
        )
    ]

    # # Create VM instance
    vm_name = f'{TEMPLATE_NAME}-{job_id}'
    print(f'Creating VM: {vm_name}')
    instance_client = compute_v1.InstancesClient()
    instance_resource = compute_v1.Instance()
    instance_resource.name = vm_name
    instance_resource.zone = ZONE
    instance_resource.machine_type = f'zones/{ZONE}/machineTypes/{MACHINE_TYPE}'
    instance_resource.service_accounts = [
        compute_v1.ServiceAccount(
            email=SERVICE_ACCOUNT_EMAIL,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    ]
    instance_resource.disks = [
        compute_v1.AttachedDisk(
            boot=True,
            auto_delete=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=f'projects/{PROJECT_ID}/global/images/{IMAGE_NAME}',
            ),
        )
    ]
    instance_resource.guest_accelerators = [
        compute_v1.AcceleratorConfig(
            accelerator_count=1,
            accelerator_type=f"zones/{ZONE}/acceleratorTypes/{GPU}",
        )
    ]
    # Disable live migration; no need for High Availability
    # We can't migrate a running job, and
    # in any case, GPU VMs don't support live migration
    instance_resource.scheduling = compute_v1.Scheduling(
        automatic_restart=False,
        on_host_maintenance="TERMINATE",
    )
    instance_resource.network_interfaces = [network_interface]  # Attach the network interface
    instance_resource.service_accounts = [
        compute_v1.ServiceAccount(
            email=SERVICE_ACCOUNT_EMAIL,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    ]
    operation = instance_client.insert(
        project=PROJECT_ID, zone=ZONE, instance_resource=instance_resource
    )
    # Wait for operation to complete
    operation.result()
    print(f'Created VM: {vm_name}')

    # Wait for VM to start
    print('Starting VM...')
    while True:
        instance_resource = instance_client.get(project=PROJECT_ID, zone=ZONE, instance=vm_name)
        if instance_resource.status == 'RUNNING':
            print('VM is running')
            # Wait for sshd to start
            print('Wait for sshd to start (60s)')
            time.sleep(60)
            break
        time.sleep(5)
        print('...still waiting for VM to start')

    # Set up VM CLI command prefix, to be used in ssh commands below
    cmd_prefix = ['gcloud', 'compute', 'ssh', '--zone', ZONE, vm_name, '--command']

    # Execute job directly (change directory and run command)
    print(f'Running job: {job_id}')
    job_command = (f'cd {JOB_DIRECTORY} && PZ_ENV=dev ./run-celery-docker.sh '
                   f'--job_config_name {job_config_name} '
                   f'--job_id {job_id} '
                   f'--batch_size {batch_size} '
                   f'--num_epochs {num_epochs} '
                   f'--num_batches {num_batches}')
    try:
        subprocess.run([*cmd_prefix, job_command], check=True)
    except Exception as ex:
        # Delete VM if we couldn't start the job
        delete_vm(instance_client, vm_name)
        raise ex

    # Wait for job completion (using file-based signal)
    print('Waiting for job completion...')
    while True:
        try:
            ssh_command = f'ls {JOB_COMPLETION_FILE}'
            result = subprocess.run(
                [*cmd_prefix, ssh_command],
                # We just care about the success or failure of the command
                # so send output to /dev/null
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            # `ls` returns `0` exit code if file is found,
            # and `1` if it isn't found
            if result.returncode == 0:
                print('Job completed.')
                break
        except Exception as e:
            print(f'Error checking for job completion, will retry: {e}')
        time.sleep(10)

    delete_vm(instance_client, vm_name)
    print('Done.')


def delete_vm(instance_client, vm_name):
    # Stop and delete VM
    print('Stopping VM...')
    operation = instance_client.stop(project=PROJECT_ID, zone=ZONE, instance=vm_name)
    operation.result()
    print('Deleting VM...')
    operation = instance_client.delete(project=PROJECT_ID, zone=ZONE, instance=vm_name)
    operation.result()
    print('VM deleted...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_config_name', required=True)
    parser.add_argument('--job_id', required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--num_epochs', type=int, required=False)
    parser.add_argument('--num_batches', type=int, required=False)
    args = parser.parse_args()

    main(
        args.job_config_name,
        args.job_id,
        args.batch_size,
        args.num_epochs,
        args.num_batches,
    )
