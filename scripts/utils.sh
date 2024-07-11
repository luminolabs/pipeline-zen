#!/bin/bash

LOCAL_ENV="local"
PROJECT_ID="neat-airport-407301"

if [[ "$PZ_ENV" == "" ]]; then
  PZ_ENV="$LOCAL_ENV"
fi

# Export .env environment variables; note, we aren't aware of which environment
# we're running on before importing PZ_ENV from .env,
# so we can't cd to /pipeline-zen-jobs conditionally above
eval $(cat ./.env | grep -v '^#' | tr -d '\r')
echo "PZ_ENV set to $PZ_ENV"

is_truthy() {
  local value=$1
  if [[ "$value" == "yes" ]] || [[ "$value" == "1" ]] || [[ "$value" == "true" ]]; then
    echo "1"
    return
  fi
  echo "0"
}

# Function to extract GPU config name from VM name
get_cluster_name_from_vm_name() {
    # Extract GPU config name from VM name;
    # ex. vm name: pipeline-zen-jobs-1xv100-us-central1-ushf -> GPU config name: 1xv100
    local vm_name=$1
    echo $vm_name | sed -E 's/pipeline-zen-jobs-([^ -]+)-[^-]+-.*/\1/'
}

# Function to extract region from MIG name
get_region_from_mig_name() {
  MIG_NAME=$1
  REGION=$(echo "$MIG_NAME" | rev | cut -d'-' -f1,2 | rev )
  echo "$REGION"
}