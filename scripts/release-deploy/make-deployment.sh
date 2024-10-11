#!/bin/bash

# Deploys a release to a new VM image and creates templates for different GPU configurations
# note: this script doesn't create a new release, it only deploys an existing release;
# the actual release process is implemented in github actions

set -e  # Exit immediately if a command fails

# Source utility functions and variables
source ./scripts/utils.sh

# Parse target environment
target_env=${1:-dev}  # default to dev if not specified
echo "Deploying to target environment: $target_env"

# New version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores

# --- Variables ---

# Service account to load to the Job VMs
JOBS_VM_SERVICE_ACCOUNT="pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
# Prefix for most resources created by this script, also used for some folder names
RESOURCES_PREFIX="pipeline-zen-jobs"
# Folder where scripts are stored
SCRIPTS_FOLDER="scripts"
# Name of the base image to use for the new image
NEW_IMAGE_NAME="${RESOURCES_PREFIX}-${VERSION_FOR_IMAGE}"
# Path to the Docker image containing the ML pipeline, to pull on the VM
DOCKER_IMAGE_REGION="us-central1"
DOCKER_IMAGE_HOST="$DOCKER_IMAGE_REGION-docker.pkg.dev"
DOCKER_IMAGE_NAME="celery-workflow"
DOCKER_IMAGE_PATH="$DOCKER_IMAGE_HOST/${PROJECT_ID}/lum-docker-images/$DOCKER_IMAGE_NAME:${VERSION}"
# Name of the VM that we will use to create the new image
IMAGE_CREATOR_VM_NAME="gha-jobs-vm-image-creator"
IMAGE_CREATOR_VM_ZONE="us-central1-a"

# GPU / CPU configurations, along with the template name to use for each
CONFIGS=(
  "count=1,type=nvidia-tesla-v100 n1-highmem-8 $RESOURCES_PREFIX-1xv100"
  "count=4,type=nvidia-tesla-v100 n1-highmem-16 $RESOURCES_PREFIX-4xv100"
  "count=8,type=nvidia-tesla-v100 n1-highmem-32 $RESOURCES_PREFIX-8xv100"
  "count=1,type=nvidia-tesla-a100 a2-highgpu-1g $RESOURCES_PREFIX-1xa100-40gb"
  "count=2,type=nvidia-tesla-a100 a2-highgpu-2g $RESOURCES_PREFIX-2xa100-40gb"
  "count=4,type=nvidia-tesla-a100 a2-highgpu-4g $RESOURCES_PREFIX-4xa100-40gb"
  "count=8,type=nvidia-tesla-a100 a2-highgpu-8g $RESOURCES_PREFIX-8xa100-40gb"
  "count=1,type=nvidia-a100-80gb a2-ultragpu-1g $RESOURCES_PREFIX-1xa100-80gb"
  "count=2,type=nvidia-a100-80gb a2-ultragpu-2g $RESOURCES_PREFIX-2xa100-80gb"
  "count=4,type=nvidia-a100-80gb a2-ultragpu-4g $RESOURCES_PREFIX-4xa100-80gb"
  "count=8,type=nvidia-a100-80gb a2-ultragpu-8g $RESOURCES_PREFIX-8xa100-80gb"
  "count=8,type=nvidia-h100-80gb a3-highgpu-8g $RESOURCES_PREFIX-8xh100-80gb"
)

# --- Main Script ---

echo "Starting deployment process..."

# Start VM
echo "Starting VM..."
gcloud compute instances start $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Wait for machine to be ready, give it a few seconds
echo "Wait 60s to allow VM to start services..."
sleep 60

echo "Pull latest code from git..."
stty -echo  # Hide the user input, so the password is not displayed
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command="git config --global --add safe.directory /$RESOURCES_PREFIX"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command="cd /$RESOURCES_PREFIX && ssh-agent bash -c \"ssh-add ~/.ssh/id_rsa; git -c core.sshCommand='ssh -o StrictHostKeyChecking=no' pull\""
stty echo  # Restore the user input

echo "Copying .env file to VM..."
gcloud compute scp ./deploy-artifacts/$target_env.env $IMAGE_CREATOR_VM_NAME:/$RESOURCES_PREFIX/.env --zone $IMAGE_CREATOR_VM_ZONE

# Grab older Docker Image IDs
old_image_id=$(gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker image ls -q")

# Build, tag, and push the Docker image
echo "Building the Docker image; version $VERSION..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker build -f celery.Dockerfile -t $DOCKER_IMAGE_NAME:local ."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker tag $DOCKER_IMAGE_NAME:local $DOCKER_IMAGE_PATH"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker push $DOCKER_IMAGE_PATH"

# Remove old Docker Images
#echo "Removing old Docker images..."
#gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker image rm -f $old_image_id || true"

# Pull Docker Image on VM
#echo "Pulling new Docker image on VM: $VERSION..."
#gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "docker pull $DOCKER_IMAGE_PATH"

# Stop VM
echo "Stopping VM..."
gcloud compute instances stop $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Create Image from VM Disk
echo "Creating new VM image..."
gcloud compute images create $NEW_IMAGE_NAME --source-disk $IMAGE_CREATOR_VM_NAME --source-disk-zone $IMAGE_CREATOR_VM_ZONE

# Create new compute instance templates for each configuration
echo "Creating new compute instance templates..."
for config in "${CONFIGS[@]}"; do
  IFS=' ' read -r accelerator machine_type template_suffix <<< "$config"
  new_template_name="${template_suffix}-${VERSION_FOR_IMAGE}"
  echo "Creating template: $new_template_name"
  gcloud compute instance-templates create $new_template_name \
    --project=$PROJECT_ID \
    --machine-type=$machine_type \
    --accelerator=$accelerator \
    --service-account=$JOBS_VM_SERVICE_ACCOUNT \
    --metadata=startup-script=/$RESOURCES_PREFIX/$SCRIPTS_FOLDER/mig-runtime/startup-script.sh \
    --create-disk=auto-delete=yes,boot=yes,device-name=$new_template_name,image=projects/$PROJECT_ID/global/images/$NEW_IMAGE_NAME,mode=rw,size=2000,type=pd-balanced \
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

# TODO: Add a step to delete the old image and templates

echo "New VM image and templates created. Deployment process complete!"
