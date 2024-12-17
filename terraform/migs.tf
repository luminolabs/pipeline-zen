locals {
  version = trimsuffix(replace(file("${path.module}/../VERSION"), ".", "-"), "\n")

  # Machine type mappings
  machine_types = {
    # V100 machine types
    "1xv100" = {
      "machine_type" = "n1-highmem-8",
      "accelerator" = {
        "type"  = "nvidia-tesla-v100",
        "count" = 1
      }
    },
    "4xv100" = {
      "machine_type" = "n1-highmem-16",
      "accelerator" = {
        "type"  = "nvidia-tesla-v100",
        "count" = 4
      }
    },
    "8xv100" = {
      "machine_type" = "n1-highmem-32",
      "accelerator" = {
        "type"  = "nvidia-tesla-v100",
        "count" = 8
      }
    },

    # A100-40GB machine types
    "1xa100-40gb" = {
      "machine_type" = "a2-highgpu-1g",
      "accelerator" = {
        "type"  = "nvidia-tesla-a100",
        "count" = 1
      }
    },
    "2xa100-40gb" = {
      "machine_type" = "a2-highgpu-2g",
      "accelerator" = {
        "type"  = "nvidia-tesla-a100",
        "count" = 2
      }
    },
    "4xa100-40gb" = {
      "machine_type" = "a2-highgpu-4g",
      "accelerator" = {
        "type"  = "nvidia-tesla-a100",
        "count" = 4
      }
    },
    "8xa100-40gb" = {
      "machine_type" = "a2-highgpu-8g",
      "accelerator" = {
        "type"  = "nvidia-tesla-a100",
        "count" = 8
      }
    },

    # A100-80GB machine types
    "1xa100-80gb" = {
      "machine_type" = "a2-ultragpu-1g",
      "accelerator" = {
        "type"  = "nvidia-a100-80gb",
        "count" = 1
      }
    },
    "2xa100-80gb" = {
      "machine_type" = "a2-ultragpu-2g",
      "accelerator" = {
        "type"  = "nvidia-a100-80gb",
        "count" = 2
      }
    },
    "4xa100-80gb" = {
      "machine_type" = "a2-ultragpu-4g",
      "accelerator" = {
        "type"  = "nvidia-a100-80gb",
        "count" = 4
      }
    },
    "8xa100-80gb" = {
      "machine_type" = "a2-ultragpu-8g",
      "accelerator" = {
        "type"  = "nvidia-a100-80gb",
        "count" = 8
      }
    },

    # H100-80GB machine types
    "8xh100-80gb" = {
      "machine_type" = "a3-highgpu-8g",
      "accelerator" = {
        "type"  = "nvidia-h100-80gb",
        "count" = 8
      }
    }
  }

  # Region configurations
  regions = {
    # V100 regions
    v100 = {
      "us-central1" = ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"]
    }

    # A100-40GB regions
    a100_40gb = {
      "asia-northeast1" = ["asia-northeast1-a", "asia-northeast1-c"]
      "asia-northeast3" = ["asia-northeast3-a", "asia-northeast3-b"]
      "asia-southeast1" = ["asia-southeast1-b", "asia-southeast1-c"]
      "europe-west4" = ["europe-west4-a", "europe-west4-b"]
      "me-west1" = ["me-west1-c"]
      "us-central1" = ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"]
      "us-east1" = ["us-east1-b"]
      "us-west1" = ["us-west1-b"]
      "us-west3" = ["us-west3-b"]
      "us-west4" = ["us-west4-b"]
    }

    # A100-80GB regions
    a100_80gb = {
      "us-central1" = ["us-central1-a", "us-central1-c"]
      "us-east4" = ["us-east4-c"]
      "us-east5" = ["us-east5-b"]
      "asia-southeast1" = ["asia-southeast1-c"]
    }

    # H100-80GB regions
    h100_80gb = {
      "us-central1" = ["us-central1-a", "us-central1-c"]
    }
  }

  # GPU configurations per type
  gpu_configs = {
    v100 = ["1xv100", "4xv100", "8xv100"]
    a100_40gb = ["1xa100-40gb", "2xa100-40gb", "4xa100-40gb", "8xa100-40gb"]
    a100_80gb = ["1xa100-80gb", "2xa100-80gb", "4xa100-80gb", "8xa100-80gb"]
    h100_80gb = ["8xh100-80gb"]
  }

  # Generate all MIG configurations
  mig_configs_temp = flatten([
    for gpu_type, configs in local.gpu_configs : [
      for region, zones in local.regions[gpu_type] : [
        for config in configs : {
          key = "pipeline-zen-jobs-${config}-${region}"
          value = {
            gpu_type     = gpu_type
            config       = config
            region       = region
            zones        = zones
            machine_type = local.machine_types[config]["machine_type"]
          }
        }
      ]
    ]
  ])

  # Convert to map format
  mig_configs = {
    for item in local.mig_configs_temp : item.key => item.value
  }
}

resource "google_compute_region_instance_group_manager" "pipeline_zen_jobs" {
  for_each = local.mig_configs

  project = var.project_id
  region  = each.value.region

  name               = "pipeline-zen-jobs-${each.value.config}-${each.value.region}-mig"
  base_instance_name = "pipeline-zen-jobs-${each.value.config}-${each.value.region}-vm"
  target_size        = 0

  version {
    instance_template = google_compute_instance_template.pipeline_zen_jobs[each.value.config].id
  }

  distribution_policy_target_shape = "EVEN"
  distribution_policy_zones        = each.value.zones

  update_policy {
    instance_redistribution_type = "NONE"
    minimal_action               = "NONE"
    type                         = "OPPORTUNISTIC"
    max_surge_fixed              = length(each.value.zones)
    max_unavailable_fixed        = length(each.value.zones)
  }

  standby_policy {
    mode = "MANUAL"
  }

  instance_lifecycle_policy {
    force_update_on_repair    = "NO"
    default_action_on_failure = "DO_NOTHING"
  }
}