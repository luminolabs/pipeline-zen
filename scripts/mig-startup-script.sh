#!/bin/bash

# Loaded onto MIG VMs to run the Pub/Sub job listener on startup

set -e  # Exit immediately if a command fails

# Go to the /pipeline-zen-jobs directory, where we've loaded all necessary files to run the ML pipeline
cd /pipeline-zen-jobs || { echo "Failed to change directory to /pipeline-zen-jobs"; exit 1; }

# Export .env environment variables
export $(grep -v '^#' ./.env | xargs)

# Define the log file path
LOG_FILE="./output.log"

# Call the pubsub-job-runner.sh script and redirect stdout and stderr to the log file
./scripts/pubsub-job-runner.sh > "$LOG_FILE" 2>&1

# Delete the VM after the script finishes;
# also removes the VM from the MIG
python ./scripts/delete_vm.py >> "$LOG_FILE" 2>&1

# Copy output.log to the only folder under .logs
LOGS_DIR=$(find .logs -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -d "$LOGS_DIR" ]; then
  cp "$LOG_FILE" "$LOGS_DIR/"
else
  echo "No directory found under .logs"
fi