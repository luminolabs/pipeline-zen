#!/bin/bash

# Update MIGs with new templates
#
# We keep this process outside of the `make-deployment.sh` script to ensure that:
# - live MIGs are not affected during the deployment process.
# - updating MIGs requires a lot of permissions, so it's better to run this process manually.

set -e  # Exit immediately if a command fails

source ./scripts/utils.sh

# Function to update a single MIG
update_mig() {
    local mig_name=$1
    local version=$2
    # Extract the region from the MIG name
    mig_region=$(get_region_from_mig_name $mig_name)
    # Extract the template prefix from the MIG name
    template_prefix=$(echo $mig_name | sed 's/-[a-z]*-[a-z]*[0-9]*$//')
    # Construct the new template name
    new_template_name="${template_prefix}-${version}"
    echo "Updating MIG: $mig_name to use template: $new_template_name"
    # Update the MIG with the new template
    gcloud compute instance-groups managed set-instance-template $mig_name \
      --region=$mig_region \
      --template=$new_template_name > /dev/null 2>&1
}

# Parse command line arguments
MIG_PREFIX=""
VERSION=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--mig) MIG_PREFIX="$2"; shift ;;
        -v|--version) VERSION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# If version is not provided, read it from VERSION file
if [ -z "$VERSION" ]; then
    if [ -f VERSION ]; then
        VERSION=$(cat VERSION)
    else
        echo "Error: VERSION file not found and no version provided."
        exit 1
    fi
fi

# Replace dots with hyphens in the version
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-')

echo "Using version: $VERSION"

if [ -n "$MIG_PREFIX" ]; then
    echo "Updating MIGs starting with: $MIG_PREFIX"
    # Get the list of MIGs that start with the specified prefix
    migs=$(gcloud compute instance-groups managed list --format="csv[no-heading](name)" --filter="name~^$MIG_PREFIX")
    if [ -z "$migs" ]; then
        echo "No MIGs found starting with: $MIG_PREFIX"
        exit 0
    fi
else
    echo "Updating all MIGs with new templates..."
    # Get the list of all MIGs
    migs=$(gcloud compute instance-groups managed list --format="csv[no-heading](name)")
fi

# Loop through each MIG and update it with the new template
while IFS=',' read -r mig_name; do
    update_mig "$mig_name" "$VERSION_FOR_IMAGE" &
done <<< "$migs"

# Wait for all background commands to finish
wait

echo "MIG update process completed."