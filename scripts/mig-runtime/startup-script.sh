#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

echo "MIG startup script started."

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || echo "Failed to change directory to /pipeline-zen-jobs - assuming local environment"

# Create directories for logs and results
mkdir -p .results
mkdir -p .logs

# Import shared utility functions
source ./scripts/utils.sh

# Define the log file path
log_file="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr
exec > >(tee -a "$log_file") 2>&1

# Export secrets
source ./scripts/mig-runtime/export-secrets.sh

# Call the pubsub-listener.sh script
echo "Starting Pub/Sub listener..."
./scripts/mig-runtime/pubsub-listener.sh

# Don't try to delete a VM if running locally, because there is no VM to delete
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  # Whether to allow the VM to continue to run after job completion
  # This flag is set by the pubsub-listener.sh script
  keep_alive=$(cat .results/.keep_alive || echo "false")
  if [[ $(is_truthy "$keep_alive") == "1" ]]; then
    echo "keep_alive flag is truthy. Skipping VM deletion."
  else
    echo "Initiating VM deletion..."
    vm_name=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    vm_zone=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
    gcloud compute instances delete $vm_name --zone=vm_zone --quiet
  fi
else
  echo "Running locally. Skipping VM deletion."
fi

# Cleanup the temporary files
rm -f .results/.keep_alive .results/.job_id

echo "MIG startup script ended."