#!/bin/bash

# Deploys a release to a new VM image and creates templates for different GPU configurations
# note: this script doesn't create a new release, it only deploys an existing release;
# the actual release process is implemented in github actions
# Steps taken by this script:
# 1. Define variables and configurations needed for the deployment.
# 2. Start the VM used for creating the new image.
# 3. Wait for the VM to initialize and start necessary services.
# 4. Copy necessary files to the VM.
# 5. Install required Python dependencies on the VM.
# 6. Remove any older Docker images from the VM.
# 7. Pull the latest version Docker image onto the VM.
# 8. Stop the VM after preparations are complete.
# 9. Create a new VM image from the VM disk.
# 10. Generate new compute instance templates for each specified GPU/CPU configuration.
# 11. Update the MIGs with the new templates.
# 12. Indicate completion of the deployment process.

set -e  # Exit immediately if a command fails

source ./scripts/utils.sh

# New version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores

# --- Variables ---

# Service account to load to the Job VMs
JOBS_VM_SERVICE_ACCOUNT="pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
# Prefix for most resources created by this script, also used for some folder names
RESOURCES_PREFIX="pipeline-zen-jobs"
# Name of the base image to use for the new image
NEW_IMAGE_NAME="${RESOURCES_PREFIX}-${VERSION_FOR_IMAGE}"
# Path to the Docker image containing the ML pipeline, to pull on the VM
DOCKER_IMAGE_REGION="us-central1"
DOCKER_IMAGE_HOST="$DOCKER_IMAGE_REGION-docker.pkg.dev"
DOCKER_IMAGE_PATH="$DOCKER_IMAGE_HOST/${PROJECT_ID}/lum-docker-images/celery-workflow:${VERSION}"
# Name of the VM that we will use to create the new image
IMAGE_CREATOR_VM_NAME="gha-jobs-vm-image-creator"
IMAGE_CREATOR_VM_ZONE="us-central1-a"

# GPU / CPU configurations, along with the template name to use for each
CONFIGS=(
  "count=1,type=nvidia-tesla-v100 n1-highcpu-8 $RESOURCES_PREFIX-1xv100"
  "count=1,type=nvidia-tesla-a100 a2-highgpu-1g $RESOURCES_PREFIX-1xa100-40gb"
  "count=4,type=nvidia-tesla-a100 a2-highgpu-4g $RESOURCES_PREFIX-4xa100-40gb"
  "count=8,type=nvidia-tesla-a100 a2-highgpu-8g $RESOURCES_PREFIX-8xa100-40gb"
  "count=16,type=nvidia-tesla-a100 a2-megagpu-16g $RESOURCES_PREFIX-16xa100-40gb"
  "count=1,type=nvidia-a100-80gb a2-highgpu-1g $RESOURCES_PREFIX-1xa100-80gb"
  "count=4,type=nvidia-a100-80gb a2-highgpu-4g $RESOURCES_PREFIX-4xa100-80gb"
  "count=8,type=nvidia-a100-80gb a2-highgpu-8g $RESOURCES_PREFIX-8xa100-80gb"
)

# --- Main Script ---

echo "Starting deployment process..."

# Start VM
echo "Starting VM..."
gcloud compute instances start $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Wait for machine to be ready, give it a few seconds
echo "Wait 60s to allow VM to start services..."
sleep 60

echo "Copying files to VM..."
# Make sure we have access permissions to the files and folders
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "sudo chown -R $(whoami):$(whoami) /$RESOURCES_PREFIX"
# Remove old files and create scripts folder
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE \
  --command "rm -rf /$RESOURCES_PREFIX/scripts /$RESOURCES_PREFIX/VERSION /$RESOURCES_PREFIX/.__pylibs__ && mkdir -p /$RESOURCES_PREFIX/scripts"
# Copy files to VM
gcloud compute scp VERSION $IMAGE_CREATOR_VM_NAME:/$RESOURCES_PREFIX --zone $IMAGE_CREATOR_VM_ZONE
gcloud compute scp ./scripts/*.py ./scripts/*.sh ./scripts/*.txt $IMAGE_CREATOR_VM_NAME:/$RESOURCES_PREFIX/scripts --zone $IMAGE_CREATOR_VM_ZONE
# TODO: Choose the right .env file based on the environment
gcloud compute scp ./deploy-artifacts/dev.env $IMAGE_CREATOR_VM_NAME:/$RESOURCES_PREFIX/.env --zone $IMAGE_CREATOR_VM_ZONE

# Install python dependencies
echo "Installing python dependencies..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "pip install --target=/$RESOURCES_PREFIX/.__pylibs__ -Ur /$RESOURCES_PREFIX/scripts/requirements.txt"

# Grab older Docker Image ID
OLD_IMAGE_ID=$(gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker image ls -q")
# Remove older Docker Image
echo "Deleting older VM image..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker image rm -f $OLD_IMAGE_ID || true"

# Pull Docker Image on VM
echo "Pulling new Docker image on VM: $VERSION..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "gcloud auth configure-docker $DOCKER_IMAGE_HOST --quiet"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker pull $DOCKER_IMAGE_PATH"

# Stop VM
echo "Stopping VM..."
gcloud compute instances stop $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Create Image from VM Disk
echo "Creating new VM image..."
gcloud compute images create $NEW_IMAGE_NAME --source-disk $IMAGE_CREATOR_VM_NAME --source-disk-zone $IMAGE_CREATOR_VM_ZONE

# Create new compute instance templates for each configuration
echo "Creating new compute instance templates..."
for CONFIG in "${CONFIGS[@]}"; do
  IFS=' ' read -r ACCELERATOR MACHINE_TYPE TEMPLATE_SUFFIX <<< "$CONFIG"
  NEW_TEMPLATE_NAME="${TEMPLATE_SUFFIX}-${VERSION_FOR_IMAGE}"
  echo "Creating template: $NEW_TEMPLATE_NAME"
  gcloud compute instance-templates create $NEW_TEMPLATE_NAME \
    --project=$PROJECT_ID \
    --machine-type=$MACHINE_TYPE \
    --accelerator=$ACCELERATOR \
    --service-account=$JOBS_VM_SERVICE_ACCOUNT \
    --metadata=startup-script=/$RESOURCES_PREFIX/scripts/mig-startup-script.sh \
    --create-disk=auto-delete=yes,boot=yes,device-name=$NEW_TEMPLATE_NAME,image=projects/$PROJECT_ID/global/images/$NEW_IMAGE_NAME,mode=rw,size=2000,type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --no-shielded-secure-boot \
    --no-shielded-vtpm \
    --no-shielded-integrity-monitoring \
    --reservation-affinity=none \
    --network-interface=network=default,network-tier=PREMIUM \
    --scopes=https://www.googleapis.com/auth/cloud-platform > /dev/null 2>&1 &
done
# Wait for all background commands to finish
wait

# Update MIGs with new templates
echo "Updating MIGs with new templates..."
# Get the list of MIGs
MIGS=$(gcloud compute instance-groups managed list --format="csv[no-heading](name)")
# Loop through each MIG and update it with the new template
while IFS=',' read -r MIG_NAME; do
  (
    # Extract the region from the MIG name
    MIG_REGION=$(get_region_from_mig_name $MIG_NAME)
    # Extract the template prefix from the MIG name
    TEMPLATE_PREFIX=$(echo $MIG_NAME | sed 's/-[a-z]*-[a-z]*[0-9]*$//')
    # Construct the new template name
    NEW_TEMPLATE_NAME="${TEMPLATE_PREFIX}-${VERSION_FOR_IMAGE}"
    echo "Updating MIG: $MIG_NAME to use template: $NEW_TEMPLATE_NAME"
    # Update the MIG with the new template
    gcloud compute instance-groups managed set-instance-template $MIG_NAME \
      --region=$MIG_REGION \
      --template=$NEW_TEMPLATE_NAME > /dev/null 2>&1
  ) &
done <<< "$MIGS"
# Wait for all background commands to finish
wait

# TODO: Add a step to delete the old image and templates

echo "New VM image and templates created, MIGs updated. Deployment process complete!"
