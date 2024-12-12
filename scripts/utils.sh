#!/bin/bash

#########################
### Environment Setup ###
#########################

LOCAL_ENV="local"
PROJECT_ID="neat-airport-407301"
SERVICE_ACCOUNT="pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"

if [[ "$PZ_ENV" == "" ]]; then
  PZ_ENV="$LOCAL_ENV"
fi

# Export .env environment variables; note, we aren't aware of which environment
# we're running on before importing PZ_ENV from .env,
# so we can't cd to /pipeline-zen-jobs conditionally above
set -o allexport
eval $(cat ./.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

########################
### Helper Functions ###
########################

is_truthy() {
  local value=$1
  if [[ "$value" == "yes" ]] || [[ "$value" == "1" ]] || [[ "$value" == "true" ]]; then
    echo "1"
    return
  fi
  echo "0"
}

# Function to extract cluster name from VM name
# ex. pipeline-zen-jobs-4xa100-40gb-us-west4-t9sj -> 4xa100-40gb
get_cluster_name_from_vm_name() {
    echo "$1" | sed -E 's/^pipeline-zen-jobs-//; s/-[^-]+-[^-]+-[^-]+$//'
}

# Function to extract region from MIG name
# ex. pipeline-zen-jobs-4xa100-40gb-us-west4-t9sj -> us-west4
get_region_from_mig_name() {
  MIG_NAME=$1
  REGION=$(echo "$MIG_NAME" | rev | cut -d'-' -f1,2 | rev )
  echo "$REGION"
}

########################
### Shared Variables ###
########################

# Service account to load to the Job VMs
JOBS_VM_SERVICE_ACCOUNT="pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
# Prefix for most resources created by this script, also used for some folder names
RESOURCES_PREFIX="pipeline-zen-jobs"
# Folder where scripts are stored
SCRIPTS_FOLDER="scripts"
# Name of the base image to use for the new image
NEW_IMAGE_NAME="${RESOURCES_PREFIX}-${VERSION_FOR_IMAGE}"

# GPU / CPU configurations, along with the template name to use for each
CONFIGS=(
  "count=1,type=nvidia-tesla-v100 n1-highmem-8 $RESOURCES_PREFIX-1xv100"
  "count=4,type=nvidia-tesla-v100 n1-highmem-16 $RESOURCES_PREFIX-4xv100"
  "count=8,type=nvidia-tesla-v100 n1-highmem-32 $RESOURCES_PREFIX-8xv100"
  "count=1,type=nvidia-tesla-a100 a2-highgpu-1g $RESOURCES_PREFIX-1xa100-40gb"
  "count=2,type=nvidia-tesla-a100 a2-highgpu-2g $RESOURCES_PREFIX-2xa100-40gb"
  "count=4,type=nvidia-tesla-a100 a2-highgpu-4g $RESOURCES_PREFIX-4xa100-40gb"
  "count=8,type=nvidia-tesla-a100 a2-highgpu-8g $RESOURCES_PREFIX-8xa100-40gb"
  "count=1,type=nvidia-a100-80gb a2-ultragpu-1g $RESOURCES_PREFIX-1xa100-80gb"
  "count=2,type=nvidia-a100-80gb a2-ultragpu-2g $RESOURCES_PREFIX-2xa100-80gb"
  "count=4,type=nvidia-a100-80gb a2-ultragpu-4g $RESOURCES_PREFIX-4xa100-80gb"
  "count=8,type=nvidia-a100-80gb a2-ultragpu-8g $RESOURCES_PREFIX-8xa100-80gb"
  "count=8,type=nvidia-h100-80gb a3-highgpu-8g $RESOURCES_PREFIX-8xh100-80gb"
)