#!/bin/bash

# Creates a new Jobs VM image with updated:
# - ./scripts and VERSION
# - latest Docker Image

set -e  # Exit immediately if a command fails

# GCP Project Information
GCP_PROJECT_ID="neat-airport-407301"
GCP_REGION="us-central1"
VM_NAME="ubuntu-1xv100-vasilis"
VM_ZONE="us-central1-a"
NEW_IMAGE_BASE_NAME="ubuntu-pipeline-zen-jobs"  # Base name without the version suffix

# Version Information (pulled from VERSION file locally)
VERSION=$(cat VERSION)
VERSION_FOR_IMAGE=$(echo "$VERSION" | tr '.' '-') # Replace dots with underscores
NEW_IMAGE_NAME="${NEW_IMAGE_BASE_NAME}-${VERSION_FOR_IMAGE}"
DOCKER_IMAGE_PATH="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/lum-docker-images/train_evaluate-workflow:${VERSION}"


# --- Functions ---

function log {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $1"
}

# --- Main Script ---

log "Starting deployment and VM image update process..."

# Start VM
log "Starting VM..."
gcloud compute instances start $VM_NAME --zone $VM_ZONE

# Copy Files to VM
log "Copying files to VM..."
gcloud compute scp --recurse ./scripts VERSION $VM_NAME:~/pipeline-zen-jobs --zone $VM_ZONE

# Install python dependencies
log "Installing python dependencies..."
gcloud compute ssh $VM_NAME --zone $VM_ZONE --command "pip install -Ur /pipeline-zen-jobs/scripts/requirements.txt"

# Delete previous Docker Image
log "Deleting older VM image..."
gcloud compute ssh $VM_NAME --zone $VM_ZONE --command "docker image rm $(docker image ls -q) || true"

# Pull Docker Image on VM
log "Pulling new Docker image on VM: $VERSION..."
gcloud compute ssh $VM_NAME --zone $VM_ZONE --command "docker pull $DOCKER_IMAGE_PATH"

# Stop VM
log "Stopping VM..."
gcloud compute instances stop $VM_NAME --zone $VM_ZONE

# Create Image from VM Disk
log "Creating new VM image..."
gcloud compute images create $NEW_IMAGE_NAME --source-disk $VM_NAME --source-disk-zone $VM_ZONE

log "Deployment and VM image update completed successfully!"
