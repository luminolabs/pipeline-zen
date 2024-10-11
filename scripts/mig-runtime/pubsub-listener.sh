#!/bin/bash

set -e  # Exit immediately if a command fails

# Import utility functions
source ./scripts/utils.sh

echo "Pub/Sub job listener started."

# Set environment name and subscription ID
subscription_id_suffix="$LOCAL_ENV"  # running locally will listen to a local subscription ID
vm_name=$(uname -n)
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  subscription_id_suffix=$(get_cluster_name_from_vm_name "$vm_name")
fi
subscription_id="pipeline-zen-jobs-start-$subscription_id_suffix"
echo "Subscription ID set to $subscription_id"

# Function to send heartbeat and status update
send_heartbeat() {
  local status="$1"
  local job_id=$(cat .results/.job_id)
  local message="{\"job_id\":\"$job_id\",\"status\":\"$status\",\"vm_name\":\"$vm_name\"}"
  echo "Sending heartbeat: $message"
  gcloud pubsub topics publish pipeline-zen-jobs-heartbeats --message="$message" --project="$PROJECT_ID" > /dev/null 2>&1
}

# Function to check for stop signal
check_stop_signal() {
  local job_id=$(cat .results/.job_id)
  local response=$(gcloud pubsub subscriptions pull --project="$PROJECT_ID" pipeline-zen-jobs-stop-main --format="json" --limit=1)
  local ack_id=$(echo "$response" | jq -r '.[0].ackId')
  if [[ "$response" != "[]" ]]; then
    local stop_job_id=$(echo "$response" | jq -r '.[0].message.data' | base64 --decode | jq -r '.job_id')
    if [[ "$stop_job_id" == "$job_id" ]]; then
      echo "Stop signal received for this job."
      gcloud pubsub subscriptions ack --project="$PROJECT_ID" --ack-ids="$ack_id" pipeline-zen-jobs-stop-main > /dev/null 2>&1
      return 0
    else
      # Negative acknowledgement to return the message to the queue
      echo "Stop signal received for another job - ignoring."
      gcloud beta pubsub subscriptions modify-message-ack-deadline --project="$PROJECT_ID" --ack-ids="$ack_id" --ack-deadline=0 pipeline-zen-jobs-stop-main > /dev/null 2>&1
    fi
  fi
  return 1
}

# Function to process message, i.e., run workflow
run_workflow() {
  local message_data="$1"

  echo "Processing message: $message_data"

  # Decode the message data
  job=$(echo "$message_data" | jq -r '.')
  job_id=$(echo "$job" | jq -r '.job_id')
  workflow=$(echo "$job" | jq -r '.workflow')
  override_env=$(echo "$job" | jq -r '.override_env')  # Override the environment if specified in the message
  args=$(echo "$job" | jq -r '.args | to_entries | map("--\(.key) \(.value | tostring)") | join(" ")')

  # Override the environment if specified in the message
  if [[ "$override_env" != "null" ]]; then
    # Copy the override environment file to .env
    echo "Overriding environment variables with $override_env.env"
    cp ./deploy-artifacts/$override_env.env .env
    # Reload the environment variables
    source ./scripts/utils.sh
  fi

  # Save job_id to a file
  echo "$job_id" > .results/.job_id

  # Extract the keep_alive flag as a file
  keep_alive=$(echo "$job" | jq -r '.keep_alive')
  echo "$keep_alive" > .results/.keep_alive

  # Let the scheduler know that we found a VM;
  # The scheduler will detach the VM from the MIG so that
  # it doesn't get deleted by the MIG scaler while the job is running
  send_heartbeat "FOUND_VM"
  # Sleep for 1 minute to allow the scheduler to detach the VM
  sleep 60

  # Run the workflow script
  echo "Running workflow script..."
  echo "Workflow: $workflow"
  echo "Arguments: $args"
  ./scripts/runners/celery-wf-docker.sh $workflow $args &

  workflow_pid=$!

  while true; do
    if ! kill -0 $workflow_pid 2>/dev/null; then
      echo "Workflow process has finished."
      break
    fi

    if check_stop_signal; then
      echo "Stopping workflow process..."
      kill $workflow_pid
      # Check if the workflow process is running and wait for it to stop
      if ps -p $workflow_pid > /dev/null; then
          echo "Wait workflow process to stop..."
          wait $workflow_pid
          echo "Workflow process stopped..."
      else
          echo "Workflow process already stopped."
      fi
      send_heartbeat "STOPPED"
      echo "Sending STOPPED heartbeat..."
      return
    fi

    send_heartbeat "RUNNING"

    sleep 10  # Send heartbeat every 10 seconds
  done

  # Check for .finished file
  if [ -f ".results/.finished" ]; then
    send_heartbeat "COMPLETED"
    echo "Job completed successfully."
  else
    send_heartbeat "FAILED"
    echo "Job failed. Check logs for details."
  fi
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
  echo "false" > .results/.keep_alive
  echo "No messages to process."
fi

echo "Pub/Sub job listener finished."