#!/bin/bash

# This script is used to pull jobs from a Pub/Sub subscription and process them.
# Steps:
# 1. Set and validate environment variables.
# 2. Check if the script is running on the VM image creator and exit if true.
# 3. Set environment and subscription ID based on the PZ_ENV environment variable.
# 4. Process a message from the Pub/Sub subscription by decoding the message data and running the workflow.
# 5. Delete the VM if not running locally, which also scales down the MIG the VM is running on

set -e  # Exit immediately if a command fails

source ./scripts/utils.sh

echo "Pub/Sub job listener started."

# Set environment name and subscription ID
env="$LOCAL_ENV"
SUBSCRIPTION_ID="$LOCAL_ENV"  # running locally will listen to the `local` subscription ID
if [[ "$PZ_ENV" != "" && "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  env=$PZ_ENV
  VM_NAME=$(uname -n)
  SUBSCRIPTION_ID=$(get_subscription_id_from_vm_name "$VM_NAME")
  echo "Environment set to $env, Subscription ID set to $SUBSCRIPTION_ID."
fi

# Function to process message, i.e., run workflow
run_workflow() {
  local message_data="$1"

  echo "Processing message: $message_data"

  # Decode the message data
  job=$(echo "$message_data" | jq -r '.')
  workflow=$(echo "$job" | jq -r '.workflow')
  args=$(echo "$job" | jq -r '.args | to_entries | map("--\(.key) \(.value | tostring)") | join(" ")')

  echo "Workflow: $workflow, Args: $args"

  # Run the workflow script
  echo "Running workflow script..."
  source ./scripts/export-secrets.sh && ./scripts/run-celery-docker.sh $workflow $args
}

echo "Pulling one message from subscription $SUBSCRIPTION_ID..."
RESPONSE=$(gcloud pubsub subscriptions pull --project="$PROJECT_ID" "$SUBSCRIPTION_ID" --format="json" --limit=1 --auto-ack)

# Check if we received any message
if [[ "$RESPONSE" != "[]" ]]; then
  echo "Message received. Extracting data..."
  # Extract the message data and ack_id
  MESSAGE_DATA=$(echo "$RESPONSE" | jq -r '.[0].message.data' | base64 --decode)

  # Process the message
  run_workflow "$MESSAGE_DATA"
else
  echo "No messages to process."
fi

echo "Script completed."