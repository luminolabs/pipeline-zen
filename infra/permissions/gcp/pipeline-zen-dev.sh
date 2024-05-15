# Allow storing results to `lum-pipeline-zen` bucket only
# 1. Assign Storage Admin
gcloud storage buckets add-iam-policy-binding gs://lum-pipeline-zen \
  --member=serviceAccount:pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/storage.objectAdmin
#etag: CAg=
#kind: storage#policy
#resourceId: projects/_/buckets/lum-pipeline-zen
#version: 1

# 2a. Create new role for listing buckets
gcloud iam roles create bucket_lister \
  --project neat-airport-407301 \
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
gcloud projects add-iam-policy-binding neat-airport-407301 \
  --member=serviceAccount:pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=projects/neat-airport-407301/roles/bucket_lister
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: projects/neat-airport-407301/roles/bucket_lister
#etag: BwYYc7HNeJk=
#version: 3

# Bigquery write access to `pipeline_zen` dataset only
# NOTE: needs `bq.json present, see below
bq update \
--source bq.json \
neat-airport-407301:pipeline_zen
#Dataset 'neat-airport-407301:pipeline_zen' successfully updated.

#bq.json contents
#.......
#{
#  "access": [
#    {
#      "role": "WRITER",
#      "specialGroup": "projectWriters"
#    },
#    {
#      "role": "WRITER",
#      "userByEmail": "pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com"
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
#      "userByEmail": "pipeline-zen-dev@neat-airport-407301.iam.gserviceaccount.com"
#    }
#  ],
#  "creationTime": "1714347117639",
#  "datasetReference": {
#    "datasetId": "pipeline_zen",
#    "projectId": "neat-airport-407301"
#  },
#  "etag": "CJIdupQnEMpnAebFqW5csw==",
#  "id": "neat-airport-407301:pipeline_zen",
#  "isCaseInsensitive": false,
#  "kind": "bigquery#dataset",
#  "lastModifiedTime": "1715724490522",
#  "location": "us-central1",
#  "maxTimeTravelHours": "168",
#  "selfLink": "https://bigquery.googleapis.com/bigquery/v2/projects/neat-airport-407301/datasets/pipeline_zen",
#  "storageBillingModel": "LOGICAL",
#  "type": "DEFAULT"
#}