import argparse
import os
import subprocess
import time

from google.cloud import compute_v1

# Configuration (adjust as needed)
PROJECT_ID = 'neat-airport-407301'
ZONE = 'us-central1'
TEMPLATE_NAME = 'ubuntu-1xv100-pipeline-zen-jobs'
IMAGE_NAME = 'ubuntu-pipeline-zen-jobs'
MACHINE_TYPE = 'n1-highcpu-8'
JOB_DIRECTORY = '/pipeline-zen-jobs'
JOB_COMPLETION_FILE = os.path.join(JOB_DIRECTORY, '/.finished')


def main(job_config_name, job_id, batch_size, num_epochs, num_batches):
    vm_name = f'{TEMPLATE_NAME}-{job_id}'

    # Create VM instance
    print(f'Creating VM: {vm_name}')
    instance_client = compute_v1.InstancesClient()
    instance_resource = compute_v1.Instance()
    instance_resource.name = vm_name
    instance_resource.zone = ZONE
    instance_resource.machine_type = f'zones/{ZONE}/machineTypes/{MACHINE_TYPE}'
    instance_resource.disks = [
        compute_v1.AttachedDisk(
            boot=True,
            auto_delete=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=f'projects/{PROJECT_ID}/global/images/{TEMPLATE_NAME}',
            ),
        )
    ]
    operation = instance_client.insert(
        project=PROJECT_ID, zone=ZONE, instance_resource=instance_resource
    )
    # Wait for operation to complete
    operation.result()
    print(f'Created VM: {vm_name}')

    # Wait for VM to start
    while True:
        print('Starting VM...')
        instance_resource = instance_client.get(project=PROJECT_ID, zone=ZONE, instance=vm_name)
        if instance_resource.status == compute_v1.Instance.Status.RUNNING:
            print('VM is running')
            break
        time.sleep(5)
        print('...still waiting for VM to start')

    # Set up VM CLI command prefix
    cmd_prefix = ['gcloud', 'compute', 'ssh', '--zone', ZONE, vm_name, '--command']

    # Execute job directly (change directory and run command)
    print(f'Running job: {job_id}')
    job_command = f'cd {JOB_DIRECTORY} && bash job_script.sh {job_config_name} {job_id} {batch_size} {num_epochs} {num_batches}'
    subprocess.run([*cmd_prefix, job_command])

    # Wait for job completion (using file-based signal)
    print('Waiting for job completion...')
    while True:
        try:
            ssh_command = f'cd {JOB_DIRECTORY} && ls {JOB_COMPLETION_FILE}'
            result = subprocess.run(
                [*cmd_prefix, ssh_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print('Job completed.')
                break
        except Exception as e:
            print(f'Error checking for job completion: {e}')
        time.sleep(10)

    # Stop and delete VM
    print('Deleting VM...')
    operation = instance_client.stop(project=PROJECT_ID, zone=ZONE, instance=vm_name)
    operation.result()
    operation = instance_client.delete(project=PROJECT_ID, zone=ZONE, instance=vm_name)
    operation.result()

    print('Job completed and VM deleted.')


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
