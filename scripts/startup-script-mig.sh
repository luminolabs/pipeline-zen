#!/bin/bash

###########################
# This script is used to pull jobs from a Pub/Sub subscription and process them.
###########################

echo "$(date): Script execution started."

if [[ "$(uname -n)" == "gha-jobs-vm-image-creator" ]]; then
  echo "$(date): This script is not meant to run on the VM image creator, exiting."
  exit 1
fi

env="local"
SUBSCRIPTION_ID="local"
if [[ "$PZ_ENV" != "" && "$PZ_ENV" != "local" ]]; then
  env=$PZ_ENV
  SUBSCRIPTION_ID="$(uname -n | sed 's/-[^-]*-[^-]*-[^-]*$//')"
  echo "$(date): Environment set to $env, Subscription ID set to $SUBSCRIPTION_ID."
  cd /pipeline-zen-jobs || { echo "$(date): Failed to change directory to /pipeline-zen-jobs"; exit 1; }
fi

# Set variables
PROJECT_ID="neat-airport-407301"

echo "$(date): Project ID set to $PROJECT_ID."

# Process message, ie. run workflow
run_workflow() {
  local message_data="$1"

  echo "$(date): Processing message: $message_data"

  # Decode the message data
  job=$(echo "$message_data" | jq -r '.')
  workflow=$(echo "$job" | jq -r '.workflow')
  args=$(echo "$job" | jq -r '.args | to_entries | map("--\(.key) \(.value | tostring)") | join(" ")')

  echo "$(date): Workflow: $workflow, Args: $args"

  # Run the workflow script
  echo "$(date): Running workflow script..."
  source ./scripts/export-secrets.sh && ./scripts/run-celery-docker.sh $workflow $args 2>&1 | tee out.log
}

echo "$(date): Pulling one message from subscription $SUBSCRIPTION_ID..."
RESPONSE=$(gcloud pubsub subscriptions pull --project="$PROJECT_ID" "$SUBSCRIPTION_ID" --format="json" --limit=1 --auto-ack)

# Check if we received any message
if [[ "$RESPONSE" != "[]" ]]; then
  echo "$(date): Message received. Extracting data..."
  # Extract the message data and ack_id
  MESSAGE_DATA=$(echo "$RESPONSE" | jq -r '.[0].message.data' | base64 --decode)

  # Process the message
  run_workflow "$MESSAGE_DATA"
else
  echo "$(date): No messages to process."
fi

echo "$(date): Script completed."

# Delete VM when done
if [[ "$env" != "local" ]]; then
  echo "$(date): Deleting VM..."
  ./scripts/delete-vm.sh
  echo "$(date): ...VM deleted"
fi
