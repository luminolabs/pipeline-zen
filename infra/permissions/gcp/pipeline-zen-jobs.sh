# Description: Permissions for job VMs

ENV="dev"
PROJECT_ID="neat-airport-407301"

# Service account for job VMs
SERVICE_ACCOUNT="serviceAccount:pipeline-zen-jobs-$ENV@$PROJECT_ID.iam.gserviceaccount.com"
# Docker image repo info
DOCKER_IMAGE_REGION="us-central1"
DOCKER_IMAGE_REPO="lum-docker-images"
# Results bucket for storing pipeline results
RESULTS_BUCKET="lum-pipeline-zen"

# Allow storing results to `lum-pipeline-zen` bucket only
# 1. Assign Storage Admin
gcloud storage buckets add-iam-policy-binding gs://$RESULTS_BUCKET \
  --member=SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin

# 2a. Create new role for listing buckets
gcloud iam roles create bucket_lister \
  --project $PROJECT_ID \
  --title "Bucket Lister" \
  --description "Grants permission to list Cloud Storage buckets." \
  --permissions storage.buckets.list,storage.buckets.get

# 2b. Assign Bucket Lister
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=projects/$PROJECT_ID/roles/bucket_lister

# Allow writes to logging service
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/logging.logWriter

# Bigquery write access to `pipeline_zen` dataset only
# NOTE: needs `bq.json present, see below
bq update \
--source bq.json \
$PROJECT_ID:pipeline_zen

# Allow access to GCP secrets such as `huggingface_token`
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/secretmanager.secretAccessor
gcloud beta projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/secretmanager.viewer

# Allow access to sending/receiving from Pub/Sub
# TODO: Need to narrow down to specific subscriptions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/pubsub.subscriber
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/pubsub.publisher

# 1a. Create new role for deleting VMs from MIGs
gcloud iam roles create mig_instance_deleter \
  --project=$PROJECT_ID \
  --title="MIG Instance Deleter" \
  --description="Grants permission to delete MIG instances." \
  --permissions=compute.instanceGroupManagers.update,compute.instances.delete

# 1b. Assign MIG Instance Deleter
# TODO: Need to narrow down to specific MIGs, and VMs, and include all regions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=projects/$PROJECT_ID/roles/mig_instance_deleter

# 2. Assign Compute Viewer, needed by the gcloud library
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role=roles/compute.viewer