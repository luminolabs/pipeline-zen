LOCAL_ENV="local"
PROJECT_ID="neat-airport-407301"

# Function to extract subscription ID from VM name
function get_subscription_id_from_vm_name() {
    # Extract subscription ID from VM name;
    # ex. vm name: pipeline-zen-jobs-1xv100-us-central1-ushf -> subscription ID: pipeline-zen-jobs-1xv100
    local vm_name=$1
    echo $vm_name | sed "s/-[^-]*-[^-]*-[^-]*$//"
}
