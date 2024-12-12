locals {
  buckets = {
    us = {
      name_suffix = "us"
      location = "US"
    }
    asia = {
      name_suffix = "asia"
      location = "ASIA"
    }
    europe = {
      name_suffix = "europe"
      location = "EU"
    }
    me_west1 = {
      name_suffix = "me-west1"
      location = "ME-WEST1"
    }
  }
}

resource "google_storage_bucket" "pipeline_zen" {
  for_each = local.buckets

  name          = "lum-${var.environment}-pipeline-zen-jobs-${each.value.name_suffix}"
  location      = each.value.location

  versioning {
    enabled = false
  }
}