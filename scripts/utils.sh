#!/bin/bash

# Function to check if a string value is truthy
is_truthy() {
  local value=$1
  if [[ "$value" == "yes" ]] || [[ "$value" == "1" ]] || [[ "$value" == "true" ]]; then
    echo "1"
    return
  fi
  echo "0"
}

# Function to extract cluster name from VM name
# ex. pipeline-zen-jobs-4xa100-40gb-us-west4-vm-t9sj -> 4xa100-40gb
get_cluster_name_from_vm_name() {
    echo "$1" | sed -E 's/^pipeline-zen-jobs-//; s/-[^-]+-[^-]+-[^-]+-[^-]+$//'
}

# The name of the local environment
LOCAL_ENV="local"
IS_GCP="0"

# See if .gcp file exists, which indicates we're running on GCP
if [ -f ./.gcp ]; then
  IS_GCP="1"
  cd /pipeline-zen-jobs || exit 0
fi

if [[ $(is_truthy "$IS_GCP") == "1" ]]; then
  PZ_ENV=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/PZ_ENV")
else
  PZ_ENV=LOCAL_ENV
fi

# If we're running locally, export variables from the .env file
if [[ $(is_truthy "$IS_GCP") == "0" ]]; then
  set -o allexport
  eval $(cat ./.env | grep -v '^#' | tr -d '\r')
fi

# Set the project ID and service account based on the environment
PROJECT_ID="eng-ai-$PZ_ENV"
SERVICE_ACCOUNT="pipeline-zen-jobs-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Work with the correct service account in local environment
export CLOUDSDK_CORE_ACCOUNT=$SERVICE_ACCOUNT

# Echo variables for debugging
echo "Current directory: $(pwd)"
echo "IS_GCP set to $IS_GCP"
echo "PZ_ENV set to $PZ_ENV"
echo "PROJECT_ID set to $PROJECT_ID"
echo "SERVICE_ACCOUNT set to $SERVICE_ACCOUNT"