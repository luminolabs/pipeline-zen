#!/bin/bash

# This script is used to pull jobs from a Pub/Sub subscription and process them.
# Steps:
# 1. Set and validate environment variables.
# 2. Check if the script is running on the VM image creator and exit if true.
# 3. Set environment and subscription ID based on the PZ_ENV environment variable.
# 4. Process a message from the Pub/Sub subscription by decoding the message data and running the workflow.
# 5. Delete the VM if not running locally, which also scales down the MIG the VM is running on

set -e  # Exit immediately if a command fails

# Import utility functions
source ./scripts/utils.sh

echo "Pub/Sub job listener started."

# Set environment name and subscription ID
subscription_id="$LOCAL_ENV"  # running locally will listen to the `local` subscription ID
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  vm_name=$(uname -n)
  subscription_id=$(get_subscription_id_from_vm_name "$vm_name")
fi
echo "Subscription ID set to $subscription_id"

# Function to process message, i.e., run workflow
run_workflow() {
  local message_data="$1"

  echo "Processing message: $message_data"

  # Decode the message data
  job=$(echo "$message_data" | jq -r '.')
  workflow=$(echo "$job" | jq -r '.workflow')
  args=$(echo "$job" | jq -r '.args | to_entries | map("--\(.key) \(.value | tostring)") | join(" ")')

  # Extract the keep_alive flag as a file
  # This file will be used by the mig-startup-script.sh script to determine whether to delete the VM
  # after the job is completed
  keep_alive=$(echo "$job" | jq -r '.keep_alive')
  echo "$keep_alive" > .keep_alive

  # Run the workflow script
  echo "Running workflow script..."
#  source ./scripts/export-secrets.sh && ./scripts/run-celery-docker.sh $workflow $args
}

echo "Pulling one message from subscription $subscription_id..."
response=$(gcloud pubsub subscriptions pull --project="$PROJECT_ID" "$subscription_id" --format="json" --limit=1 --auto-ack)

# Check if we received any message
if [[ "$response" != "[]" ]]; then
  echo "Message received. Extracting data..."
  # Extract the message data and ack_id
  message_data=$(echo "$response" | jq -r '.[0].message.data' | base64 --decode)

  # Process the message
  run_workflow "$message_data"
else
  echo "No messages to process."
fi

echo "Pub/Sub job listener finished."
