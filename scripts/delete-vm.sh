#!/bin/bash

PROJECT_ID='neat-airport-407301'
ZONE_URL="http://metadata.google.internal/computeMetadata/v1/instance/zone"
UNAME_CMD="uname -n"
METADATA_FLAVOR="Metadata-Flavor: Google"

# Function to get the VM name using uname
get_vm_name_from_uname() {
    echo "Executing \`uname -n\` to get the VM name..."
    vm_name=$(uname -n)
    echo "VM name obtained: ${vm_name}"
    echo "${vm_name}"
}

# Function to get the zone of the VM from the metadata server
get_zone() {
    echo "Fetching the zone from the metadata server..."
    zone=$(curl -s -H "${METADATA_FLAVOR}" "${ZONE_URL}")
    zone=${zone##*/}
    echo "Zone obtained: ${zone}"
    echo "${zone}"
}

# Function to delete the instance from the MIG
delete_instance_from_mig() {
    vm_name=$1
    vm_zone=$2
    mig_name=$(echo "${vm_name}" | sed 's/-[^-]*-[^-]*-[^-]*$//')

    echo "Deleting instance: ${vm_name} from MIG: ${mig_name} in zone: ${vm_zone} for project: ${PROJECT_ID}..."
    gcloud compute instance-groups managed delete-instances "${mig_name}" --instances="${vm_name}" --zone="${vm_zone}" --project="${PROJECT_ID}" --quiet
    echo "Instance ${vm_name} deleted from MIG ${mig_name} successfully."
}

main() {
    vm_name=$(get_vm_name_from_uname)
    vm_zone=$(get_zone)
    echo "VM name - ${vm_name}, VM zone - ${vm_zone}"

    delete_instance_from_mig "${vm_name}" "${vm_zone}"
}

main
