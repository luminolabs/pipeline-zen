# Allows downloading docker image from `lum-docker-images` repo only
gcloud artifacts repositories add-iam-policy-binding --location us-central1 lum-docker-images \
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=roles/artifactregistry.reader
#Updated IAM policy for repository [lum-docker-images].
#bindings:
#- members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/artifactregistry.reader
#etag: BwYYcRQkDhE=
#version: 1

# Allow storing results to `lum-pipeline-zen` bucket only
# 1. Assign Storage Admin
gcloud storage buckets add-iam-policy-binding gs://lum-pipeline-zen \
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/storage.objectAdmin
#etag: CAc=
#kind: storage#policy
#resourceId: projects/_/buckets/lum-pipeline-zen
#version: 1

# 2a. Create new role for listing buckets
gcloud iam roles update bucket_lister \
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
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=projects/neat-airport-407301/roles/bucket_lister
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: projects/neat-airport-407301/roles/bucket_lister

# Allow writes to logging service
gcloud projects add-iam-policy-binding neat-airport-407301 \
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=roles/logging.logWriter
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#...
#...
#- members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/logging.logWriter
#etag: BwYYY8cu1Bc=
#version: 1

# Allow deleting Job VMs with name that
# starts with `ubuntu-1xv100-pipeline-zen-jobs-` only
# ie. only Job VMs
# 1. Create new role for deleting VMs
gcloud iam roles update compute_instance_deleter \
  --project=neat-airport-407301 \
  --title="Compute Instance Deleter" \
  --description="Grants permission to delete Compute Engine instances." \
  --permissions=compute.instances.delete
#Created role [compute_instance_deleter].
#description: Grants permission to delete Compute Engine instances.
#etag: BwYYcclc9s0=
#includedPermissions:
#- compute.instances.delete
#name: projects/neat-airport-407301/roles/compute_instance_deleter
#stage: ALPHA
#title: Compute Instance Deleter

# 2. Assign Compute Instance Deleter
gcloud projects add-iam-policy-binding neat-airport-407301 \
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=projects/neat-airport-407301/roles/compute_instance_deleter --condition=expression="resource.name.startsWith('projects/neat-airport-407301/zones/us-central1-a/instances/ubuntu-1xv100-pipeline-zen-jobs-')",title='ml-pipeline-job-vms',description="ML Pipeline Job VMs"
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#- condition:
#    description: ML Pipeline Job VMs
#    expression: resource.name.startsWith('projects/neat-airport-407301/zones/us-central1-a/instances/ubuntu-1xv100-pipeline-zen-jobs-')
#    title: ml-pipeline-job-vms
#  members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: projects/neat-airport-407301/roles/compute_instance_deleter
#...
#...
#etag: BwYYY_IqtDg=
#version: 3

# 3. Assign Compute Viewer, needed by the gcloud library
gcloud projects add-iam-policy-binding neat-airport-407301 \
  --member=serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com \
  --role=roles/compute.viewer
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#- members:
#  - serviceAccount:pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com
#  role: roles/compute.viewer
#...
#...

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
#      "userByEmail": "pipeline-zen-jobs-dev@neat-airport-407301.iam.gserviceaccount.com"
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