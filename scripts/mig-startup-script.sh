#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
# cd /pipeline-zen-jobs || { echo "Failed to change directory to /pipeline-zen-jobs"; exit 1; }

# Export .env environment variables
export $(grep -v '^#' ./.env | xargs)

# Define the log file path
LOG_FILE="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr using tee
exec > >(tee -a "$LOG_FILE") 2>&1

# Call the pubsub-job-runner.sh script
./scripts/pubsub-job-runner.sh

./scripts/utils.sh

PZ_KEEP_ALIVE=$(cat .keep_alive)

# Check PZ_KEEP_ALIVE before deleting the VM
if $(is_truthy $PZ_KEEP_ALIVE); then
  # Delete the VM after the script finishes; also removes the VM from the MIG
  python ./scripts/delete_vm.py
fi

echo "PZ_KEEP_ALIVE is set to $PZ_KEEP_ALIVE. Skipping VM deletion."

rm -rf .keep_alive
