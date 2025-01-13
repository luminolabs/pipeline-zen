locals {
  # Other terraform projects reference these buckets, so if you change this config,
  # make sure to update the other projects as well.
  # - scheduler-zen
  # - lumino-api
  # -
  env_buckets = {
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
  }
  local_buckets = var.environment == "dev" ? {
    local = {
      name_suffix = "us"
      location    = "US"
      environment = "local"
    }
  } : {}
  buckets = merge(
    local.env_buckets,
    local.local_buckets
  )
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
  for_each = toset(concat([var.environment],
    var.environment == "dev" ? ["local"] : []))

  name     = "lum-${each.key}-pipeline-zen-datasets"
  location = "US"

  versioning {
    enabled = false
  }
}