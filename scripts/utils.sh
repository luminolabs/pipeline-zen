LOCAL_ENV="local"
PROJECT_ID="neat-airport-407301"

is_truthy() {
  VAR=$1
  if [[ "$VAR" == "yes" ]] || [[ "$VAR" == "1" ]] || [[ "$VAR" == "true" ]]; then
    echo "0"
  fi
  echo "1"
}

# Function to extract subscription ID from VM name
function get_subscription_id_from_vm_name() {
    # Extract subscription ID from VM name;
    # ex. vm name: pipeline-zen-jobs-1xv100-us-central1-ushf -> subscription ID: pipeline-zen-jobs-1xv100
    local vm_name=$1
    echo $vm_name | sed "s/-[^-]*-[^-]*-[^-]*$//"
}

# Function to extract region from MIG name
get_region_from_mig_name() {
  MIG_NAME=$1
  REGION=$(echo "$MIG_NAME" | rev | cut -d'-' -f1,2 | rev )
  echo "$REGION"
}