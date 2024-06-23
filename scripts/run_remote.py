#!python

"""
This script is used to run a workflow using DWS MIGs (Managed Instance Groups).

It performs the following steps:
1. Parse command line arguments to get the workflow, target MIG, job configuration name, and job ID.
2. Generate a unique job ID if one is not provided.
3. Resize the target MIG by increasing its size by one instance.
4. Publish a message to a Pub/Sub topic with the workflow details and arguments, for the VMs in the MIG to consume.
"""

import argparse
import uuid
import json
import logging

from google.cloud import pubsub_v1, compute_v1

from utils import PROJECT_ID, get_region_from_mig_name, get_subscription_id_from_mig_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Initialize a Publisher client
publisher = pubsub_v1.PublisherClient()

# Define the Pub/Sub topic
topic_id = 'pipeline-zen-jobs'
topic_path = publisher.topic_path(PROJECT_ID, topic_id)


def resize_mig(mig_name: str, region: str, project_id: str):
    """
    Resize the Managed Instance Group (MIG) by increasing its size by one.

    :param mig_name: The name of the MIG to resize
    :param region: The region of the MIG
    :param project_id: The project ID of the MIG
    """
    client = compute_v1.RegionInstanceGroupManagersClient()

    # Get the current size of the MIG
    mig = client.get(project=project_id, region=region, instance_group_manager=mig_name)
    current_size = mig.target_size
    new_size = current_size + 1

    # Resize the MIG
    operation = client.resize(project=project_id, region=region, instance_group_manager=mig_name, size=new_size)

    # Wait for operation to complete
    operation.result()
    logging.info(f'MIG {mig_name} resized from {current_size} to {new_size}')


def publish_message(message: dict, subscription_name: str):
    """
    Publish a message to the Pub/Sub topic.

    :param message: The message to publish
    :param subscription_name: The target GPU configuration to use for training; used as the subscription name
    """
    future = publisher.publish(topic_path, json.dumps(message).encode('utf-8'), **{'mig': subscription_name})
    logging.info(f'Published message ID: {future.result()}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the model training workflow remotely')
    parser.add_argument('-wf', '--workflow', type=str, required=True,
                        help='The workflow to run; ex. `torchtunewrapper`, `train_evaluate`')
    parser.add_argument('-tm', '--mig_name', type=str, required=True,
                        help='The target MIG to use for training')
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help='The name of the job config file, without the `.py` extension')
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help='The job ID to use for training; logs and other job results and '
                             'artifacts will be named after this.')

    # Store the rest of the arguments as unknown arguments
    known_args, unknown_args = parser.parse_known_args()
    workflow, mig_name, job_config_name, job_id = (
        known_args.workflow, known_args.mig_name, known_args.job_config_name, known_args.job_id)

    # Create auto-generated job ID if one is not given
    job_id = job_id or (job_config_name + '-' + str(uuid.uuid4()))

    # Get the subscription ID from the MIG name
    subscription_id = get_subscription_id_from_mig_name(mig_name)

    # Create the message to be sent
    message = {
        'workflow': workflow,
        'args': {
            **{
                'job_config_name': job_config_name,
                'job_id': job_id,
            }, **{
                # Add all other unknown CLI args to the message
                k.replace('--', ''): v for k, v in zip(unknown_args[::2], unknown_args[1::2])
            }
        }
    }

    if mig_name == 'local':
        logging.info('Skipping MIG resize and publishing message on local subscription ID for local consumption...')
    else:
        logging.info(f'Running workflow `{workflow}` on MIG `{mig_name}` with job ID `{job_id}`...')
        # Resize the target MIG to allow for the new job to run
        resize_mig(mig_name, get_region_from_mig_name(mig_name), PROJECT_ID)

    # Publish the message to the Pub/Sub topic
    publish_message(message, subscription_id)
    logging.info('Workflow scheduled successfully!')
