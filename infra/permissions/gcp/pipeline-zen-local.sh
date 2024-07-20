# Description: Permissions for local development

ENV="dev"
PROJECT_ID="neat-airport-407301"

# Service account for local development
SERVICE_ACCOUNT="serviceAccount:pipeline-zen-local-$ENV@$PROJECT_ID.iam.gserviceaccount.com"

# Results bucket for storing pipeline results
RESULTS_BUCKET="lum-pipeline-zen"

# Allow storing results to `lum-pipeline-zen-*` buckets only
# 1. Assign Storage Admin
gcloud storage buckets add-iam-policy-binding gs://$RESULTS_BUCKET-us \
  --member=$SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding gs://$RESULTS_BUCKET-asia \
  --member=$SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding gs://$RESULTS_BUCKET-europe \
  --member=$SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding gs://$RESULTS_BUCKET-west1 \
  --member=$SERVICE_ACCOUNT \
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

# Bigquery write access to `pipeline_zen` dataset only
# NOTE: needs `bq-policy.json` present, and edited with new permissions
# ie. bq show --format=prettyjson $PROJECT_ID:pipeline_zen > bq-policy.json
bq update \
--source bq-policy.json \
$PROJECT_ID:pipeline_zen