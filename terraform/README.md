# Pipeline Zen Infrastructure

This directory contains the Terraform configuration for Pipeline Zen's cloud infrastructure on Google Cloud Platform (GCP). 

The infrastructure is designed to support scalable machine learning job processing across multiple regions and GPU types.

## Core Components

### Managed Instance Groups (MIGs)
- Defined in `migs.tf`
- Dynamic configuration for different GPU types and regions
- Supports multiple GPU configurations:
    - V100 (1x, 4x, 8x)
    - A100-40GB (1x, 2x, 4x, 8x)
    - A100-80GB (1x, 2x, 4x, 8x)
    - H100-80GB (8x)
- Regional distribution with zone redundancy
- Zero-sized by default, scaled up on demand

### Storage
- Defined in `storage.tf`
- Multi-regional buckets for job results:
    - US
    - Asia
    - Europe
    - Middle East (ME-WEST1)
- Dataset storage buckets
- Consistent naming conventions using environment prefixes

### Pub/Sub Topics & Subscriptions
- Defined in `pubsub.tf`
- Job control and monitoring topics:
    - Start jobs (`pipeline-zen-jobs-start`)
    - Stop jobs (`pipeline-zen-jobs-stop`)
    - Heartbeats (`pipeline-zen-jobs-heartbeats`)
    - Job metadata (`pipeline-zen-jobs-meta`)
- Per-cluster subscriptions with filtering

### Instance Templates
- Defined in `templates.tf`
- Version-controlled templates for each GPU configuration
- Pre-configured with:
    - Boot disk using custom VM image
    - Network configuration
    - GPU accelerator setup
    - Service account association
    - Startup scripts

### IAM & Permissions
- Defined in `permissions.tf`
- Custom service account for job execution
- Custom IAM roles:
    - Bucket listing permissions
    - VM deletion permissions
- Granular bucket access controls
- Pub/Sub access management

## Design Patterns

### 1. Dynamic Resource Generation
- Uses Terraform locals and map structures for configuration
- Generates resources based on configuration maps
- Easy to add new GPU types or regions

### 2. Environment Segregation
- Separate configurations for different environments (dev, prod)
- Environment-specific variable files
- Consistent resource naming with environment prefixes

### 3. Secret Management
- Secrets stored in Secret Manager
- Configuration loaded from environment files
- Separate handling of Terraform vs application secrets

### 4. Resource Naming Conventions
```
{service}-{purpose}-{specs}-{region}-{resource-type}
Example: pipeline-zen-jobs-8xa100-40gb-us-central1-mig
```

## Setup and Usage

### Prerequisites
1. Install OpenTofu
2. GCP project access and credentials
3. Required environment files:
    - `{env}-config.env` - Application configuration for target environment
    - `secrets.tfvars` - Terraform secrets, same for all environments
    - `{env}.tfvars` - Environment variables for target environment

### Configuration Files
1. Copy `config-example.env` to `{env}-config.env`
2. Copy `secrets-example.tfvars` to `secrets.tfvars`
3. Set required variables in both files

### Deploy Infrastructure
```bash
# Initialize Terraform
tofu init

# Plan changes
tofu plan -var-file="{env}.tfvars" -var-file="secrets.tfvars"

# Apply changes
tofu apply -var-file="{env}.tfvars" -var-file="secrets.tfvars"
```


## Infrastructure Updates

### Version Control
- VM image versions tracked in VERSION file (previous images are maintained)
- Templates automatically versioned based on VERSION (previous templates are deleted)

### Rolling Updates
- Zero downtime deployments, existing jobs not affected

## Development Guidelines

### Making Changes
1. Define resource configurations in appropriate .tf files
2. Follow existing naming conventions
3. Update README.md with new components
4. Test in dev environment first

## Security Considerations

### Access Control
- Principle of least privilege
- Custom IAM roles for specific permissions

### Secret Management
- Application secrets are stored in Secret Manager
- Environment-specific application secrets