LOCAL_ENV="local"
PROJECT_ID="neat-airport-407301"

# Function to extract subscription ID from VM name
function get_subscription_id_from_vm_name() {
    # Extract subscription ID from VM name;
    # ex. vm name: pipeline-zen-jobs-1xv100-us-central1-ushf -> subscription ID: pipeline-zen-jobs-1xv100
    local vm_name=$1
    echo $vm_name | sed "s/-[^-]*-[^-]*-[^-]*$//"
}

is_truthy() {
  var=$1
  if [[ "$var" == "yes" ]] || [[ "$var" == "1" ]] || [[ "$var" == "true" ]]; then
    return 0
  fi
  return 1
}

# Function to extract region from MIG name
get_region_from_mig_name() {
  MIG_NAME=$1
  REGION=$(echo "$MIG_NAME" | rev | cut -d'-' -f1,2 | rev )
  echo "$REGION"
}