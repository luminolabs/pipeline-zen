# Pipeline Deployments

## Update VERSION file
Make sure the `VERSION` file is updated with the new version number. This version number will be used to tag the Docker image and name the VM image.

## Prerequisites
The release is being prepared in a VM in GCP Compute Engine. Make sure you have the following:
- Access to `gha-jobs-vm-image-creator` VM in GCP Compute Engine under the `neat-airport-407301` project
- On the VM:
  - An `~/.ssh/id_rsa` password protected private ssh key that can access your GitHub account
  - Your public key set under `~/.ssh/authroized_keys` to be able to ssh to the VM

## Create a new VM image
Go to the `pipeline-zen` root folder and run the following command:
```bash
./scripts/release-deploy/create-vm-image.sh
```
This will create a VM image in GCP Compute Engine with the latest Docker image. 

The image is named as follows: `pipeline-zen-jobs-<version>`, ex: `pipeline-zen-jobs-0-55-8`

Click [here](https://console.cloud.google.com/compute/images?tab=images&orgonly=true&project=neat-airport-407301) to confirm that the new VM image is created.

The following artifacts will be created, all with the same version in their resource name:
- New Docker image pushed to GCP Artifact Registry ([link](https://console.cloud.google.com/artifacts/docker/neat-airport-407301/us-central1/lum-docker-images/celery-workflow?orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4Pp4IBBlPFOqWRl_oXtqWz1tKogAha1A9xlkGFmznTN5MclhYMQQnWp92Z9ZxzTvSKqtb9hBxviHOlAsBUx9QuQ4Uezg3rjC_p6oupVYpjLedlyEI8))
- New VM image in GCP Compute Engine, which has the new Docker image loaded ([link](https://console.cloud.google.com/compute/images?tab=images&orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4OwY5BvEoTEpnwB6kbKhwbAPcQXsA3eDPLJF4FN0ksJYnfhinTz5Pnxr6Uip3W2nLZZrTRRLHmYIBCDpY2o37eD_o0Wwz-fqS0PL-fJsG_HnWABP8M&pli=1))
- New VM templates in GCP Compute Engine, pointing to the new VM image ([link](https://console.cloud.google.com/compute/instanceTemplates/list?orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4OClZMJm9OYWvuUB0xMgDlosfcfqub4CP9bKh4-gD3zxWyEXNXXuN65dvBGGOE-N9_BQEl6n5kvHGaMCxFW5dix5qge2MLPwwNC7PhDi--_1qg_cws))

## Deploy the new release
The rest of the deployment process is automated using terraform.

Applying the terraform scripts will:
- Create new VM templates in GCP Compute Engine, pointing to the new VM image
- Update the MIGs with the new VM templates; existing VMs and jobs won't be affected, only new jobs will use the new VM templates

Authenticate to GCP using the following command:
```bash
gcloud auth application-default login
```

Go to the `pipeline-zen` root folder, then to the `terraform` folder and run the following command:
```bash
tofu apply -var-file="<env>.tfvars" -var-file="secrets.tfvars"
```
Replace `<env>` with the environment you want to deploy to, ex: `dev`, `staging`, `prod`.

You will be prompted to review and confirm the changes. Type `yes` to apply the changes.

## Next steps
You can now [start new jobs](README.md) with the new deployment. Existing jobs will continue to run on the old deployment until they are completed.
