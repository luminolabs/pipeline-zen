#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || echo "Failed to change directory to /pipeline-zen-jobs - assuming local environment"

# Export .env environment variables; note, we aren't aware of which environment
# we're running on before importing PZ_ENV from .env,
# so we can't cd to /pipeline-zen-jobs conditionally above
export $(grep -v '^#' ./.env | xargs)

# Import shared utility functions; it has to be imported after reading the .env file
source ./scripts/utils.sh

# Define the log file path
LOG_FILE="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr
exec > >(tee -a "$LOG_FILE") 2>&1

# Call the pubsub-job-runner.sh script
./scripts/pubsub-job-runner.sh

# Don't try to delete a VM if running locally, because there is no VM to delete
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  # Whether to allow the VM to continue to run after job completion
  # This flag is set by the pubsub-job-runner.sh script
  KEEP_ALIVE=$(cat .keep_alive) || false
  if is_truthy "$KEEP_ALIVE"; then
    echo "KEEP_ALIVE flag is truthy. Skipping VM deletion."
  else
    python ./scripts/delete_vm.py
  fi
fi

# Cleanup the .keep_alive flag file
rm -rf .keep_alive || true
