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
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
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
  --member="principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/pipeline-zen" \
  --role=roles/artifactregistry.writer
#Updated IAM policy for repository [lum-docker-images].
#bindings:
#...
#...
#- members:
#  - principal://iam.googleapis.com/projects/482988686822/locations/global/workloadIdentityPools/github/subject/pipeline-zen
#  role: roles/artifactregistry.writer
#etag: BwYYc2Z8Shg=
#version: 1