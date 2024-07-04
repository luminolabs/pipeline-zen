#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

echo "MIG startup script started."

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || echo "Failed to change directory to /pipeline-zen-jobs - assuming local environment"

# Import shared utility functions
source ./scripts/utils.sh

# Define the log file path
log_file="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr
exec > >(tee -a "$log_file") 2>&1

# Call the pubsub-listener.sh script
source ./scripts/mig-runtime/export-secrets.sh && ./scripts/mig-runtime/pubsub-listener.sh

# Don't try to delete a VM if running locally, because there is no VM to delete
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  # Whether to allow the VM to continue to run after job completion
  # This flag is set by the pubsub-job-runner.sh script
  keep_alive=$(cat .keep_alive || echo "false")
  if [[ $(is_truthy "$keep_alive") == "1" ]]; then
    echo "keep_alive flag is truthy. Skipping VM deletion."
  else
    PYTHONPATH=$PYTHONPATH python ./scripts/mig-runtime/delete_vm.py
  fi
else
  echo "Running locally. Skipping VM deletion."
fi

# Cleanup the .keep_alive flag file
rm -rf .keep_alive || true

echo "MIG startup script ended."