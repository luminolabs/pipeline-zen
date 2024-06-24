#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <machine-type> <size> <template-version>"
    exit 1
fi

machine_type=$1
size=$2
template_version=$3
template_version_dash=${template_version//./-}

# Define the array of regions and their corresponding zones
regions_to_zones=(
    "asia-northeast1:asia-northeast1-a,asia-northeast1-c"
    "asia-northeast3:asia-northeast3-a,asia-northeast3-b"
    "asia-southeast1:asia-southeast1-b,asia-southeast1-c"
    "europe-west4:europe-west4-a,europe-west4-b"
    "me-west1:me-west1-b,me-west1-c"
    "us-central1:us-central1-a,us-central1-c"
    "us-east1:us-east1-a,us-east1-b"
    "us-west1:us-west1-b"
    "us-west3:us-west3-b"
    "us-west4:us-west4-b"
)

echo "Starting Managed Instance Group creation/resizing process..."

create_or_resize_mig() {
    region=$1
    zones=$2
    mig_name="pipeline-zen-jobs-${machine_type}-${region}"

    # Check if the Managed Instance Group (MIG) already exists
    mig_exists=$(gcloud beta compute instance-groups managed list --filter="name=$mig_name" --format="value(name)" 2>/dev/null)

    if [ -z "$mig_exists" ]; then
        # If the MIG does not exist, create it with the input target size
        echo "Creating Managed Instance Group: $mig_name with target size $size"
        gcloud beta compute instance-groups managed create $mig_name \
            --project=neat-airport-407301 \
            --base-instance-name=$mig_name \
            --template=projects/neat-airport-407301/global/instanceTemplates/pipeline-zen-jobs-${machine_type}-${template_version_dash} \
            --size=$size \
            --zones=$zones \
            --target-distribution-shape=EVEN \
            --instance-redistribution-type=NONE \
            --default-action-on-vm-failure=do_nothing \
            --no-force-update-on-repair \
            --standby-policy-mode=manual \
            --list-managed-instances-results=PAGELESS > /dev/null 2>&1
    else
        # If the MIG exists, get the number of running instances and the current target size
        running_instances=$(gcloud beta compute instance-groups managed list-instances $mig_name --region=$region 2>/dev/null | grep RUNNING | wc -l)
        current_target_size=$(gcloud beta compute instance-groups managed describe $mig_name --region=$region --format="value(targetSize)" 2>/dev/null)

        # Trim whitespace
        running_instances=$(echo $running_instances | xargs)
        current_target_size=$(echo $current_target_size | xargs)

        # Determine the new target size
        if [ $running_instances -gt $size ]; then
            target_size=$running_instances
        elif [ $running_instances -eq 0 ]; then
            target_size=$size
        else
            target_size=$size
        fi

        echo "Managed Instance Group $mig_name already exists with current target size $current_target_size and $running_instances running instances. New target size: $target_size"

        gcloud beta compute instance-groups managed resize $mig_name \
            --project=neat-airport-407301 \
            --region=$region \
            --size=$target_size > /dev/null 2>&1
    fi
}

# Export function and variables for parallel execution
export -f create_or_resize_mig
export machine_type
export size
export template_version_dash

# Iterate over each region and its corresponding zones in parallel
for region_zone in "${regions_to_zones[@]}"; do
    region=$(echo "$region_zone" | cut -d':' -f1)
    zones=$(echo "$region_zone" | cut -d':' -f2)
    create_or_resize_mig $region $zones &
done

# Wait for all parallel jobs to finish
wait

echo "Managed Instance Group creation/resizing process completed."
