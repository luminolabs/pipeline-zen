locals {
  buckets = {
    us = {
      name_suffix = "us"
      location    = "US"
      environment = var.environment
    }
    asia = {
      name_suffix = "asia"
      location    = "ASIA"
      environment = var.environment
    }
    europe = {
      name_suffix = "europe"
      location    = "EU"
      environment = var.environment
    }
    me_west1 = {
      name_suffix = "me-west1"
      location    = "ME-WEST1"
      environment = var.environment
    }
    local = {
      name_suffix = "us"
      location    = "US"
      environment = "local"
    }
  }
}

# Create storage buckets for pipeline zen jobs results
resource "google_storage_bucket" "pipeline_zen_results" {
  for_each = local.buckets

  name     = "lum-${each.value.environment}-pipeline-zen-jobs-${each.value.name_suffix}"
  location = each.value.location

  versioning {
    enabled = false
  }
}

# Create storage bucket for pipeline zen datasets
resource "google_storage_bucket" "pipeline_zen_datasets" {
  for_each = toset([var.environment, "local"])

  name     = "lum-${each.key}-pipeline-zen-datasets"
  location = "US"

  versioning {
    enabled = false
  }
}