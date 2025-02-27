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

# The name of the local environment
LOCAL_ENV="local"
IS_GCP="0"

# Where the source code is located
PIPELINE_ZEN_JOBS_DIR="/pipeline-zen-jobs"
# Check if the folder exists
if [ -d "$PIPELINE_ZEN_JOBS_DIR" ]; then
    cd "$PIPELINE_ZEN_JOBS_DIR" || exit 0
fi

# See if .gcp file exists, which indicates we're running on GCP
if [ -f ./.gcp ]; then
  IS_GCP="1"
fi

if [[ $(is_truthy "$IS_GCP") == "1" ]]; then  # If running on GCP, get the environment from the metadata server
  PZ_ENV=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/PZ_ENV")
elif [ -z "$PZ_ENV" ]; then  # If PZ_ENV is not set, default to local
  PZ_ENV=LOCAL_ENV
fi

# If we're running locally, export variables from the .env file
if [[ "$PZ_ENV" == "local" || "$PZ_ENV" == "cpnode" ]]; then
  set -o allexport
  eval $(cat ./.env | grep -v '^#' | tr -d '\r')
fi

# Set the project ID and service account based on the environment
PROJECT_ID="eng-ai-$PZ_ENV"
# If PROJECT_ID == "eng-ai-[local|cpnode]" change it to "eng-ai-dev"
# because we don't have eng-ai-[local|cpnode] yet
if [[ "$PZ_ENV" == "local" || "$PZ_ENV" == "cpnode" ]]; then
  PROJECT_ID="eng-ai-dev"
fi
# Set the service account to use
SERVICE_ACCOUNT="pipeline-zen-jobs-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Work with the correct service account in local environment
export CLOUDSDK_CORE_ACCOUNT=$SERVICE_ACCOUNT

# Echo variables for debugging
echo "Current directory: $(pwd)"
echo "IS_GCP set to $IS_GCP"
echo "PZ_ENV set to $PZ_ENV"
echo "PROJECT_ID set to $PROJECT_ID"
echo "SERVICE_ACCOUNT set to $SERVICE_ACCOUNT"