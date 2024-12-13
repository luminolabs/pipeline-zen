#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

echo "MIG startup script started."

# Sleep for a few seconds to allow the VM to fully start up and drivers to load
sleep 30

# Import shared utility functions
source ./scripts/utils.sh || source /pipeline-zen-jobs/scripts/utils.sh

# Create directories for logs and results
mkdir -p .results

# Define the log file path
log_file="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr
exec > >(tee -a "$log_file") 2>&1

if [[ $(is_truthy "$IS_GCP") == "1" ]]; then
  # Export secrets
  echo "Fetching secrets from Secret Manager"
  SECRET_NAME="pipeline-zen-jobs-config"
  SECRET_PAYLOAD=$(gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID)
  # Parse the secret payload and set environment variables
  set -o allexport
  eval "$SECRET_PAYLOAD"
fi

# Call the pubsub-listener.sh script
echo "Starting Pub/Sub listener..."
./scripts/mig-runtime/pubsub-listener.sh

# Don't try to delete a VM if running locally, because there is no VM to delete
if [[ $(is_truthy "$IS_GCP") == "1" ]]; then
  # Whether to allow the VM to continue to run after job completion
  # This flag is set by the pubsub-listener.sh script
  keep_alive=$(cat .results/.keep_alive || echo "false")
  if [[ $(is_truthy "$keep_alive") == "1" ]]; then
    echo "keep_alive flag is truthy. Skipping VM deletion."
  else
    echo "Initiating VM deletion..."
    vm_name=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    vm_zone=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
    gcloud compute instances delete $vm_name --zone=$vm_zone --quiet
  fi
else
  echo "Running locally. Skipping VM deletion."
fi

# Cleanup the temporary files
rm -f .results/.keep_alive

echo "MIG startup script ended."