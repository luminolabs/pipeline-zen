import argparse
import uuid

from google.cloud import pubsub_v1
import json

"""
This script is used to run a workflow using DWS MIGs.
"""

# Initialize a Publisher client
publisher = pubsub_v1.PublisherClient()

# Define the Pub/Sub topic
project_id = 'neat-airport-407301'
topic_id = 'pipeline-zen-jobs'
topic_path = publisher.topic_path(project_id, topic_id)

def publish_message(message: dict, target_mig: str):
    """
    Publish a message to the topic

    :param message: The message to publish
    :param target_mig: The target MIG to use for training
    :return: None
    """
    future = publisher.publish(topic_path, json.dumps(message).encode('utf-8'), **{'mig': target_mig})
    print(f'Published message ID: {future.result()}')

if __name__ == '__main__':
    # Parse CLI arguments; we parse the job_config_name and job_id that are shared between all workflows
    parser = argparse.ArgumentParser(description='Run the model training workflow remotely')
    parser.add_argument('-wf', '--workflow', type=str, required=True,
                        help='The workflow to run; ex. `torchtunewrapper`, `train_evaluate`')
    parser.add_argument('-tm', '--target_mig', type=str, required=True,
                        help='The target MIG to use for training')
    parser.add_argument('-jc', '--job_config_name', type=str, required=True,
                        help='The name of the job config file, without the `.py` extension')
    parser.add_argument('-jid', '--job_id', type=str, required=False,
                        help='The job_id to use for training; '
                             'logs and other job results and artifacts will be named after this.')
    # And store the rest of the arguments as unknown arguments
    known_args, unknown_args = parser.parse_known_args()
    # Extract the known arguments
    workflow, target_mig, job_config_name, job_id = (
        known_args.workflow, known_args.target_mig, known_args.job_config_name, known_args.job_id)

    # Create auto-generated job id if one is not given
    job_id = job_id or (job_config_name + '-' + str(uuid.uuid4()))

    # Message to be sent
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
    # Publish the message
    publish_message(message, target_mig)
