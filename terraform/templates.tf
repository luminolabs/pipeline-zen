resource "google_compute_instance_template" "pipeline-zen-jobs" {
  for_each = local.machine_types

  name         = "pipeline-zen-jobs-${each.key}-${local.version}-tpl"
  project      = var.project_id
  machine_type = each.value.machine_type

  disk {
    source_image = "projects/${var.resources_project_id}/global/images/pipeline-zen-jobs-${local.version}"
    auto_delete  = true
    boot         = true
    device_name  = "pipeline-zen-jobs-${each.key}-${local.version}-disk"
    mode         = "READ_WRITE"
    disk_size_gb = 2000
    disk_type    = "pd-balanced"
  }

  network_interface {
    network = "default"
    access_config {
      network_tier = "PREMIUM"
    }
  }

  service_account {
    email = google_service_account.pipeline_zen_jobs.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  guest_accelerator {
    type  = each.value.accelerator.type
    count = each.value.accelerator.count
  }

  metadata = {
    startup-script = "/pipeline-zen-jobs/scripts/mig-runtime/startup-script.sh"
  }

  scheduling {
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
    provisioning_model  = "STANDARD"
  }

  shielded_instance_config {
    enable_secure_boot          = false
    enable_vtpm                 = false
    enable_integrity_monitoring = false
  }

  reservation_affinity {
    type = "NO_RESERVATION"
  }

  lifecycle {
    create_before_destroy = true
  }
}