#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || { echo "Failed to change directory to /pipeline-zen-jobs"; exit 1; }

# Export .env environment variables
export $(grep -v '^#' ./.env | xargs)

# Define the log file path
LOG_FILE="./output.log"

# Redirect both stdout and stderr to the log file and to stdout/stderr using tee
exec > >(tee -a "$LOG_FILE") 2>&1

# Call the pubsub-job-runner.sh script
./scripts/pubsub-job-runner.sh

# Check PZ_KEEP_ALIVE before deleting the VM
if [ "$PZ_KEEP_ALIVE" != "yes" ] && [ "$PZ_KEEP_ALIVE" != "1" ] && [ "$PZ_KEEP_ALIVE" != "true" ]; then
  # Delete the VM after the script finishes; also removes the VM from the MIG
  python ./scripts/delete_vm.py
else
  echo "PZ_KEEP_ALIVE is set to $PZ_KEEP_ALIVE. Skipping VM deletion."
fi
