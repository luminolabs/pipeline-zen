ENV="dev"
PROJECT_ID="neat-airport-407301"

# Note: The results of commands are stored as comments

# Allow storing results to `lum-pipeline-zen` bucket only
# 1. Assign Storage Admin
gcloud storage buckets add-iam-policy-binding gs://lum-pipeline-zen \
  --member=serviceAccount:pipeline-zen-local-$ENV@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-local-dev@neat-airport-407301.iam.gserviceaccount.com
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/storage.objectAdmin
#etag: CAg=
#kind: storage#policy
#resourceId: projects/_/buckets/lum-pipeline-zen
#version: 1

# 2a. Create new role for listing buckets
gcloud iam roles create bucket_lister \
  --project $PROJECT_ID \
  --title "Bucket Lister" \
  --description "Grants permission to list Cloud Storage buckets." \
  --permissions storage.buckets.list,storage.buckets.get
#Created role [bucket_lister].
#description: Grants permission to list Cloud Storage buckets.
#etag: BwYYcg9-eRU=
#includedPermissions:
#- storage.buckets.get
#- storage.buckets.list
#name: projects/neat-airport-407301/roles/bucket_lister
#stage: ALPHA
#title: Bucket Lister

# 2b. Assign Bucket Lister
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:pipeline-zen-local-$ENV@$PROJECT_ID.iam.gserviceaccount.com \
  --role=projects/$PROJECT_ID/roles/bucket_lister
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-local-dev@neat-airport-407301.iam.gserviceaccount.com
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: projects/neat-airport-407301/roles/bucket_lister
#etag: BwYYc7HNeJk=
#version: 3

# Bigquery write access to `pipeline_zen` dataset only
# NOTE: needs `bq-policy.json` present, and edited with new permissions
# See below
bq update \
--source bq-policy.json \
$PROJECT_ID:pipeline_zen
#Dataset 'neat-airport-407301:pipeline_zen' successfully updated.

# bq show --format=prettyjson $PROJECT_ID:pipeline_zen > bq-policy.json
#cat bq-policy.json
#
#{
#  "access": [
#    {
#      "role": "WRITER",
#      "specialGroup": "projectWriters"
#    },
#    {
#      "role": "WRITER",
#      "userByEmail": "pipeline-zen-local-dev@neat-airport-407301.iam.gserviceaccount.com"
#    },
#    {
#      "role": "WRITER",
#      "userByEmail": "pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
#    },
#    {
#      "role": "OWNER",
#      "specialGroup": "projectOwners"
#    },
#    {
#      "role": "READER",
#      "specialGroup": "projectReaders"
#    },
#    {
#      "role": "READER",
#      "userByEmail": "pipeline-zen-local-dev@neat-airport-407301.iam.gserviceaccount.com"
#    },
#    {
#      "role": "READER",
#      "userByEmail": "pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
#    }
#  ],
#  "creationTime": "1714347117639",
#  "datasetReference": {
#    "datasetId": "pipeline_zen",
#    "projectId": "neat-airport-407301"
#  },
#  "etag": "jl/Ehp27S263jt2X0YcUfQ==",
#  "id": "neat-airport-407301:pipeline_zen",
#  "isCaseInsensitive": false,
#  "kind": "bigquery#dataset",
#  "lastModifiedTime": "1715735182114",
#  "location": "us-central1",
#  "maxTimeTravelHours": "168",
#  "selfLink": "https://bigquery.googleapis.com/bigquery/v2/projects/neat-airport-407301/datasets/pipeline_zen",
#  "storageBillingModel": "LOGICAL",
#  "type": "DEFAULT"
#}