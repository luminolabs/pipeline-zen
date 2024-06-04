# See: https://github.com/google-github-actions/auth?tab=readme-ov-file#preferred-direct-workload-identity-federation

# Why use Workload Identity Pool, instead of a Service Account?
# Using a SA requires giving that SA the `roles/iam.serviceAccountTokenCreator` role
# which applies to all other SA as well, which allows this SA to impersonate
# other SA.
# On the other hand, with WIP, GitHub can authenticate with GCP
# using OpenID Connect, which issues short lived tokens.
# We'll still grant specific permissions to the OIDC principal on
# GCP, so it'll only able to access what we want it to access

PROJECT_ID="neat-airport-407301"

# Note: The results of commands are stored as comments

# 1. Setup Workload Identity Pool
gcloud iam workload-identity-pools create "github" \
  --project=$PROJECT_ID \
  --location="global" \
  --display-name="GitHub Actions Pool"
#Created workload identity pool [github].

gcloud iam workload-identity-pools describe "github" \
  --project=$PROJECT_ID \
  --location="global" \
  --format="value(name)"
#projects/482988686822/locations/global/workloadIdentityPools/github

gcloud iam workload-identity-pools providers create-oidc "pipeline-zen" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github" \
  --display-name="GitHub Repo Provider" \
  --attribute-mapping="google.subject=assertion.repository,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'luminolabs'" \
  --issuer-uri="https://token.actions.githubusercontent.com"
#Created workload identity pool provider [pipeline-zen].

gcloud iam workload-identity-pools providers describe "pipeline-zen" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github" \
  --format="value(name)"
#projects/482988686822/locations/global/workloadIdentityPools/github/providers/pipeline-zen

# Add this step to the GitHub Actions workflow
#  - name: Set up Google Cloud SDK Authentication
#    uses: google-github-actions/auth@v2
#    with:
#      project_id: 'neat-airport-407301'
#      workload_identity_provider: 'projects/482988686822/locations/global/workloadIdentityPools/github/providers/pipeline-zen'

# 2. Allows writing docker image to `lum-docker-images` repo only
gcloud artifacts repositories add-iam-policy-binding --location us-central1 lum-docker-images \
  --member="principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/luminolabs/pipeline-zen" \
  --role=roles/artifactregistry.writer
#Updated IAM policy for repository [lum-docker-images].
#bindings:
#...
#...
#- members:
#  - principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/luminolabs/pipeline-zen
#  role: roles/artifactregistry.writer
#etag: BwYYc2Z8Shg=
#version: 1

# 3. New role to allow GHA to create a new Jobs VM image upon creating a new release
gcloud iam roles update jobs_image_creator --project $PROJECT_ID \
  --title "VM Manager for ubuntu-1xv100-vasilis" \
  --description "Manage VM ubuntu-1xv100-vasilis for automated creating of new Jobs VM Image" \
  --permissions compute.projects.get,compute.instances.start,compute.instances.stop,compute.instances.get,compute.instances.getGuestAttributes,compute.instances.setMetadata,compute.disks.useReadOnly,compute.disks.use,compute.disks.get,compute.images.create,compute.images.get,compute.globalOperations.get
#Created role [jobs_image_creator].
#description: Manage VM ubuntu-1xv100-vasilis for automated creating of new Jobs VM
#  Image
#etag: BwYYl7w29bs=
#includedPermissions:
#- compute.disks.get
#- compute.disks.use
#- compute.disks.useReadOnly
#- compute.globalOperations.get
#- compute.images.create
#- compute.images.get
#- compute.instances.get
#- compute.instances.getGuestAttributes
#- compute.instances.setMetadata
#- compute.instances.start
#- compute.instances.stop
#- compute.projects.get
#name: projects/neat-airport-407301/roles/jobs_image_creator
#stage: ALPHA
#title: VM Manager for ubuntu-1xv100-vasilis

# 4. Assign jobs_image_creator
 gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/luminolabs/pipeline-zen" \
  --role "projects/$PROJECT_ID/roles/jobs_image_creator" \
  --condition=expression="(resource.name=='projects/$PROJECT_ID/zones/us-central1-a/instances/ubuntu-1xv100-vasilis' && resource.type=='compute.googleapis.com/Instance') || resource.type!='compute.googleapis.com/Instance'",title="limit_to_jobs_vm_instance_template",description="Limit compute perms to ubuntu-1xv100-vasilis VM"
#Updated IAM policy for project [neat-airport-407301].
#bindings:
#...
#...
#- condition:
#    description: Limit compute perms to ubuntu-1xv100-vasilis VM
#    expression: (resource.name=='projects/neat-airport-407301/zones/us-central1-a/instances/ubuntu-1xv100-vasilis'
#      && resource.type=='compute.googleapis.com/Instance') || resource.type!='compute.googleapis.com/Instance'
#    title: limit_to_jobs_vm_instance_template
#  members:
#  - principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/luminolabs/pipeline-zen
#  role: projects/neat-airport-407301/roles/jobs_image_creator
#etag: BwYYlp1yW3Q=
#version: 3
