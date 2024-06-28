#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || echo "Failed to change directory to /pipeline-zen-jobs - assuming local environment"

# Export .env environment variables; note, we aren't aware of which environment
# we're running on before importing PZ_ENV from .env,
# so we can't cd to /pipeline-zen-jobs conditionally above
export $(grep -v '^#' ./.env | xargs)

# Import shared utility functions
source ./scripts/utils.sh

# Define the log file path
LOG_FILE="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr using tee
exec > >(tee -a "$LOG_FILE") 2>&1

# Call the pubsub-job-runner.sh script
./scripts/pubsub-job-runner.sh

# Whether to allow the VM to continue to run after job completion
# This flag is read and set by the pubsub-job-runner.sh script
KEEP_ALIVE=$(cat .keep_alive)

# Check KEEP_ALIVE before attempting to delete the VM;
# and don't try to delete the VM if running locally
if [[ "$PZ_ENV" != "$LOCAL_ENV" ]]; then
  if is_truthy $KEEP_ALIVE; then
    # Delete the VM after the script finishes; also removes the VM from the MIG
    python ./scripts/delete_vm.py
  fi
fi
echo "KEEP_ALIVE flag is set to $KEEP_ALIVE. Skipping VM deletion."

# Cleanup the .keep_alive flag file
rm -rf .keep_alive
