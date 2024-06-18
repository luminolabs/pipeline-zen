#!/bin/bash

###########################
# This script is used to pull jobs from a Pub/Sub subscription and process them.
###########################

if [[ "$(uname -n)" == "gha-jobs-vm-image-creator" ]]; then
  echo "This script is not meant to run on the VM image creator, exiting."
  exit 1
fi

env="local"
SUBSCRIPTION_ID="local"
if [[ "$PZ_ENV" != "" && "$PZ_ENV" != "local" ]]; then
  env=$PZ_ENV
  SUBSCRIPTION_ID="$(uname -n | sed 's/-[^-]*-[^-]*-[^-]*$//')"
  cd /pipeline-zen-jobs || { echo "Failed to change directory to /pipeline-zen-jobs"; exit 1; }
fi

# Set variables
PROJECT_ID="neat-airport-407301"
SLEEP_INTERVAL=10
ACK_DEADLINE_EXTENSION=60
WORKFLOW_SCRIPT="./scripts/run-celery-docker.sh"

# Function to extend the ack deadline;
# because the default ack deadline is only 10 minutes, we need to extend it while running the job
extend_ack_deadline() {
  local ack_id=$1
  while true; do
    echo "$(date): Extending ack deadline for ack ID: $ack_id"
    gcloud beta pubsub subscriptions modify-message-ack-deadline "$SUBSCRIPTION_ID" --ack-ids="$ack_id" --ack-deadline="$ACK_DEADLINE_EXTENSION"
    sleep "$SLEEP_INTERVAL"
  done
}

# Process message, ie. run workflow
run_workflow() {
  local message_data="$1"
  local ack_id="$2"

  echo "$(date): Processing message: $message_data"

  # Decode the message data
  job=$(echo "$message_data" | jq -r '.')
  workflow=$(echo "$job" | jq -r '.workflow')
  args=$(echo "$job" | jq -r '.args | to_entries | map("--\(.key) \(.value | tostring)") | join(" ")')

  echo "$(date): Workflow: $workflow, Args: $args"

  # Start extending the ack deadline in the background
  extend_ack_deadline "$ack_id" &
  extend_pid=$!

  # Run the workflow script
  echo "$(date): Running workflow script..."
  $WORKFLOW_SCRIPT "$workflow" $args
  script_exit_code=$?

  # Stop extending the ack deadline, when the script finishes
  echo "$(date): Stopping ack deadline extension for ack ID: $ack_id"
  kill $extend_pid

  # Acknowledge the message regardless if the script ran successfully,
  # because if it failed, it will likely fail again
  # TODO: We need to log/alert failures centrally to be able to troubleshoot them
  if [ $script_exit_code -eq 0 ]; then
    echo "$(date): Acknowledging message with ack ID: $ack_id"
  else
    echo "$(date): Workflow script failed with exit code $script_exit_code, acknowledging message with ack ID: $ack_id"
  fi
  gcloud pubsub subscriptions ack "$SUBSCRIPTION_ID" --ack-ids="$ack_id"
}

echo "$(date): Pulling one message from subscription $SUBSCRIPTION_ID..."
RESPONSE=$(gcloud pubsub subscriptions pull --project="$PROJECT_ID" "$SUBSCRIPTION_ID" --format="json" --limit=1)

# Check if we received any message
if [[ "$RESPONSE" != "[]" ]]; then
  echo "$(date): Message received. Extracting data..."
  # Extract the message data and ack_id
  MESSAGE_DATA=$(echo "$RESPONSE" | jq -r '.[0].message.data' | base64 --decode)
  ACK_ID=$(echo "$RESPONSE" | jq -r '.[0].ackId')

  # Process the message
  run_workflow "$MESSAGE_DATA" "$ACK_ID"
else
  echo "$(date): No messages to process."
fi

echo "$(date): Script completed."

# Delete VM when done
if [[ "$env" != "local" ]]; then
  echo "Deleting VM..."
  cmd="python ./scripts/delete_vm.py"
  eval "${cmd}" &>/dev/null & disown;
fi
