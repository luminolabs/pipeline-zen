#!/bin/bash


#########################
### Environment Setup ###
#########################


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