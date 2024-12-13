#!/bin/bash

# Create a new VM image for the ML pipeline, using the latest code and Docker image.
# Won't deploy the image to a MIG or link it to a VM template, it'll just create a new version of the image.

set -e  # Exit immediately if a command fails

# Source utility functions and variables
source ./scripts/utils.sh

# New version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores

# Prefix for most resources created by this script, also used for some folder names
RESOURCES_PREFIX="pipeline-zen-jobs"
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

echo "Starting VM image creation process..."

# Start VM
echo "Starting VM..."
gcloud compute instances start $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Wait for machine to be ready, give it a few seconds
echo "Wait 60s to allow VM to start services..."
sleep 60

echo "Pull latest code from git..."  # TODO: When we have tags again, we should pull the code from a specific tag
stty -echo  # Hide the user input, so the password is not displayed
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command="git config --global --add safe.directory /$RESOURCES_PREFIX"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command="cd /$RESOURCES_PREFIX && ssh-agent bash -c \"ssh-add ~/.ssh/id_rsa; git -c core.sshCommand='ssh -o StrictHostKeyChecking=no' pull\""
stty echo  # Restore the user input

# Build, tag, and push the Docker image
echo "Building the Docker image; version $VERSION..."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker build -f celery.Dockerfile -t $DOCKER_IMAGE_NAME:local ."
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker tag $DOCKER_IMAGE_NAME:local $DOCKER_IMAGE_PATH"
gcloud compute ssh $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE --command "cd /$RESOURCES_PREFIX && docker push $DOCKER_IMAGE_PATH"

# Stop VM
echo "Stopping VM..."
gcloud compute instances stop $IMAGE_CREATOR_VM_NAME --zone $IMAGE_CREATOR_VM_ZONE

# Create Image from VM Disk
echo "Creating new VM image..."
gcloud compute images create $NEW_IMAGE_NAME --source-disk $IMAGE_CREATOR_VM_NAME --source-disk-zone $IMAGE_CREATOR_VM_ZONE

echo "New VM image created. Done."
