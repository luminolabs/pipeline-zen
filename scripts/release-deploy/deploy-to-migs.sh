#!/bin/bash

# Update MIGs with new templates
#
# We keep this process outside of the `make-deployment.sh` script to ensure that:
# - live MIGs are not affected during the deployment process.
# - updating MIGs requires a lot of permissions, so it's better to run this process manually.

set -e  # Exit immediately if a command fails

source ./scripts/utils.sh

# New version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores

echo "Updating MIGs with new templates..."
# Get the list of MIGs
migs=$(gcloud compute instance-groups managed list --format="csv[no-heading](name)")
# Loop through each MIG and update it with the new template
while IFS=',' read -r mig_name; do
  (
    # Extract the region from the MIG name
    mig_region=$(get_region_from_mig_name $mig_name)
    # Extract the template prefix from the MIG name
    template_prefix=$(echo $mig_name | sed 's/-[a-z]*-[a-z]*[0-9]*$//')
    # Construct the new template name
    new_template_name="${template_prefix}-${VERSION_FOR_IMAGE}"
    echo "Updating MIG: $mig_name to use template: $new_template_name"
    # Update the MIG with the new template
    gcloud compute instance-groups managed set-instance-template $mig_name \
      --region=$mig_region \
      --template=$new_template_name > /dev/null 2>&1
  ) &
done <<< "$migs"
# Wait for all background commands to finish
wait