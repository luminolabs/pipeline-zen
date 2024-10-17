# Pipeline Deployments

## Create a new deployment
Go to the `pipeline-zen` root folder and run the following command:
```bash
./scripts/release-deploy/make-deployment.sh
```
This will create all the deployment artifacts but won't actually deploy anything.

The following artifacts will be created, all with the same version in their resource name:
- New Docker image pushed to GCP Artifact Registry ([link](https://console.cloud.google.com/artifacts/docker/neat-airport-407301/us-central1/lum-docker-images/celery-workflow?orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4Pp4IBBlPFOqWRl_oXtqWz1tKogAha1A9xlkGFmznTN5MclhYMQQnWp92Z9ZxzTvSKqtb9hBxviHOlAsBUx9QuQ4Uezg3rjC_p6oupVYpjLedlyEI8))
- New VM image in GCP Compute Engine, using the new Docker image ([link](https://console.cloud.google.com/compute/images?tab=images&orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4OwY5BvEoTEpnwB6kbKhwbAPcQXsA3eDPLJF4FN0ksJYnfhinTz5Pnxr6Uip3W2nLZZrTRRLHmYIBCDpY2o37eD_o0Wwz-fqS0PL-fJsG_HnWABP8M&pli=1))
- New VM templates in GCP Compute Engine, pointing to the new VM image ([link](https://console.cloud.google.com/compute/instanceTemplates/list?orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4OClZMJm9OYWvuUB0xMgDlosfcfqub4CP9bKh4-gD3zxWyEXNXXuN65dvBGGOE-N9_BQEl6n5kvHGaMCxFW5dix5qge2MLPwwNC7PhDi--_1qg_cws&pageState=(%22instance_templates%22:(%22s%22:%5B(%22i%22:%22creationTimestamp%22,%22s%22:%221%22),(%22i%22:%22usedByNames%22,%22s%22:%221%22),(%22i%22:%22name%22,%22s%22:%220%22)%5D,%22r%22:200))

## Deploy the new deployment
Go to the `pipeline-zen` root folder and run the following command:
```bash
./scripts/release-deploy/deploy-to-migs.sh <optional env name; default: dev> <optional version; default: reads from VERSION file>
```
This will set the new VM templates to the MIGs. Existing VMs and jobs won't be affected. Click [here](([link](https://console.cloud.google.com/compute/instanceTemplates/list?orgonly=true&project=neat-airport-407301&supportedpurview=organizationId,folder,project&rapt=AEjHL4OClZMJm9OYWvuUB0xMgDlosfcfqub4CP9bKh4-gD3zxWyEXNXXuN65dvBGGOE-N9_BQEl6n5kvHGaMCxFW5dix5qge2MLPwwNC7PhDi--_1qg_cws&pageState=(%22instance_templates%22:(%22s%22:%5B(%22i%22:%22creationTimestamp%22,%22s%22:%221%22),(%22i%22:%22usedByNames%22,%22s%22:%221%22),(%22i%22:%22name%22,%22s%22:%220%22)%5D,%22r%22:200)) to confirm that the MIGs are updated with the new VM templates.

## Next steps
You can now [start new jobs](README.md) with the new deployment. Existing jobs will continue to run on the old deployment until they are completed.