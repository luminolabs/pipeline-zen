# See: https://github.com/google-github-actions/auth?tab=readme-ov-file#preferred-direct-workload-identity-federation

# Why use Workload Identity Pool, instead of a Service Account?
# Using a SA requires giving that SA the `roles/iam.serviceAccountTokenCreator` role
# which applies to all other SA as well, which allows this SA to impersonate
# other SA.
# On the other hand, with WIP, GitHub can authenticate with GCP
# using OpenID Connect, which issues short lived tokens.
# We'll still grant specific permissions to the OIDC principal on
# GCP, so it'll only able to access what we want it to access

ENV="dev"
PROJECT_ID="neat-airport-407301"

# Workload identity Pool info
WORKLOAD_IDENTITY_POOL="github"
SERVICE_ACCOUNT="principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/$WORKLOAD_IDENTITY_POOL/subject/luminolabs/pipeline-zen"
# Docker image repo info
DOCKER_IMAGE_REGION="us-central1"
DOCKER_IMAGE_REPO="lum-docker-images"

# 1. Setup Workload Identity Pool
gcloud iam workload-identity-pools create "$WORKLOAD_IDENTITY_POOL" \
  --project=$PROJECT_ID \
  --location="global" \
  --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "pipeline-zen" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="$WORKLOAD_IDENTITY_POOL" \
  --display-name="GitHub Repo Provider" \
  --attribute-mapping="google.subject=assertion.repository,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'luminolabs'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 2. Allows writing docker image to `lum-docker-images` repo only
gcloud artifacts repositories add-iam-policy-binding --location $DOCKER_IMAGE_REGION $DOCKER_IMAGE_REPO \
  --member=$SERVICE_ACCOUNT \
  --role=roles/artifactregistry.writer

# 3a. New role to allow GHA to manage the VM for creating new Jobs VM image
gcloud iam roles create jobs_image_creator --project $PROJECT_ID \
  --title "VM Manager for gha-jobs-vm-image-creator" \
  --description "Manage VM gha-jobs-vm-image-creator for automated creating of new Jobs VM Image" \
  --permissions compute.projects.get,compute.instances.start,compute.instances.stop,compute.instances.get,compute.instances.getGuestAttributes,compute.instances.setMetadata,compute.disks.useReadOnly,compute.disks.use,compute.disks.get,compute.images.create,compute.images.get,compute.globalOperations.get

# 3b. Assign jobs_image_creator
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role "projects/$PROJECT_ID/roles/jobs_image_creator" \
  --condition=expression="(resource.name=='projects/$PROJECT_ID/zones/us-central1-a/instances/gha-jobs-vm-image-creator' && resource.type=='compute.googleapis.com/Instance') || resource.type!='compute.googleapis.com/Instance'",title="limit_to_jobs_vm_instance_template",description="Limit compute perms to gha-jobs-vm-image-creator VM"

# 4a. New role to allow GHA to create VM templates
gcloud iam roles update vm_template_creator --project $PROJECT_ID \
  --title "Creates VM templates" \
  --description "Allows creation of VM templates" \
  --permissions compute.instanceTemplates.create,compute.instanceTemplates.get,compute.networks.use,compute.disks.use

# 4b. Assign vm_template_creator
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=$SERVICE_ACCOUNT \
  --role "projects/$PROJECT_ID/roles/vm_template_creator"

# 5. Allows downloading docker image from `lum-docker-images` repo only - used to allow downloading
# docker image when building a new Jobs VM image
gcloud artifacts repositories add-iam-policy-binding --location us-central1 lum-docker-images \
  --member=serviceAccount:gha-jobs-vm-image-creator-$ENV@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/artifactregistry.reader

# 6. Allow GHA principal to access `gha-jobs-vm-image-creator` service account
# when logging into `gha-jobs-vm-image-creator` VM
gcloud iam service-accounts add-iam-policy-binding \
  gha-jobs-vm-image-creator-$ENV@$PROJECT_ID.iam.gserviceaccount.com \
  --member=$SERVICE_ACCOUNT \
  --role=roles/iam.serviceAccountUser