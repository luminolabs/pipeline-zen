#!/bin/bash

# Deploys a release to a new VM image and template
# note: this script doesn't create a new release, it only deploys an existing release;
# the actual release process is implemented in github actions
#
# Creates a new VM image and template with updated:
# - ./scripts and VERSION files
# - python dependencies
# - docker image

set -e  # Exit immediately if a command fails

# GCP
PROJECT_ID="neat-airport-407301"
REGION="us-central1"
ZONE="us-central1-a"
SERVICE_ACCOUNT="pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"

# New version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores

# The VM and VM disk are both under the same name
IMAGE_CREATOR_VM_NAME="gha-jobs-vm-image-creator"

# New VM image name
BASE_IMAGE_NAME="ubuntu-pipeline-zen-jobs"
NEW_IMAGE_NAME="${BASE_IMAGE_NAME}-${VERSION_FOR_IMAGE}"

# Which docker image to load to the new VM
DOCKER_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/lum-docker-images/celery-workflow:${VERSION}"

# GPU / CPU configurations, along with the template name to use for each
CONFIGS=(
  "count=1,type=nvidia-tesla-v100 n1-highcpu-8 pipeline-zen-jobs-1xv100"
  "count=1,type=nvidia-tesla-a100 a2-highgpu-1g pipeline-zen-jobs-1xa100-40gb"
  "count=4,type=nvidia-tesla-a100 a2-highgpu-4g pipeline-zen-jobs-4xa100-40gb"
  "count=8,type=nvidia-tesla-a100 a2-highgpu-8g pipeline-zen-jobs-8xa100-40gb"
  "count=16,type=nvidia-tesla-a100 a2-megagpu-16g pipeline-zen-jobs-16xa100-40gb"
  "count=1,type=nvidia-a100-80gb a2-highgpu-1g pipeline-zen-jobs-1xa100-80gb"
  "count=4,type=nvidia-a100-80gb a2-highgpu-4g pipeline-zen-jobs-4xa100-80gb"
  "count=8,type=nvidia-a100-80gb a2-highgpu-8g pipeline-zen-jobs-8xa100-80gb"
)

# --- Main Script ---

echo "Starting deployment process..."

# Start VM
echo "Starting VM..."
gcloud compute instances start $IMAGE_CREATOR_VM_NAME --zone $ZONE

# Wait for machine to be ready, give it a few seconds
echo "Wait 60s to allow VM to start services..."
sleep 60

echo "Copying files to VM..."
# Make sure we have access permissions to the files and folders
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "sudo chown -R $(whoami):$(whoami) /pipeline-zen-jobs"
# Remove old files
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "rm -rf /pipeline-zen-jobs/scripts /pipeline-zen-jobs/VERSION || true"
# Copy files to VM
gcloud compute scp --recurse ./scripts VERSION $IMAGE_CREATOR_VM_NAME:/pipeline-zen-jobs --zone $ZONE

# Install python dependencies
echo "Installing python dependencies..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "pip install -Ur /pipeline-zen-jobs/scripts/requirements.txt"

# Delete older Docker Image
echo "Deleting older VM image..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "docker image rm \$(docker image ls -q) || true"

# Pull Docker Image on VM
echo "Pulling new Docker image on VM: $VERSION..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $ZONE --command "docker pull $DOCKER_IMAGE_PATH"

# Stop VM
echo "Stopping VM..."
gcloud compute instances stop $IMAGE_CREATOR_VM_NAME --zone $ZONE

# Create Image from VM Disk
echo "Creating new VM image..."
gcloud compute images create $NEW_IMAGE_NAME --source-disk $IMAGE_CREATOR_VM_NAME --source-disk-zone $ZONE

# Create new compute instance templates for each configuration
echo "Creating new compute instance templates..."
for CONFIG in "${CONFIGS[@]}"; do
  IFS=' ' read -r ACCELERATOR MACHINE_TYPE TEMPLATE_SUFFIX <<< "$CONFIG"
  NEW_TEMPLATE_NAME="${TEMPLATE_SUFFIX}-${VERSION_FOR_IMAGE}"
  gcloud compute instance-templates create $NEW_TEMPLATE_NAME \
    --project=$PROJECT_ID \
    --machine-type=$MACHINE_TYPE \
    --accelerator=$ACCELERATOR \
    --service-account=$SERVICE_ACCOUNT \
    --create-disk=auto-delete=yes,boot=yes,device-name=$NEW_TEMPLATE_NAME,image=projects/$PROJECT_ID/global/images/$NEW_IMAGE_NAME,mode=rw,size=2000,type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --no-shielded-secure-boot \
    --no-shielded-vtpm \
    --no-shielded-integrity-monitoring \
    --reservation-affinity=any \
    --network-interface=network=default,network-tier=PREMIUM \
    --scopes=https://www.googleapis.com/auth/cloud-platform
done

echo "New VM image and templates created. Deployment process complete!"
