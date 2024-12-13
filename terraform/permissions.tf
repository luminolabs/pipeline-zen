resource "google_service_account" "pipeline_zen_jobs" {
  account_id   = "pipeline-zen-jobs-sa"
  display_name = "Pipeline Zen Jobs Service Account"
  description  = "Service account for running pipeline jobs"
  project      = var.project_id
}

resource "google_project_iam_custom_role" "bucket_lister" {
  role_id     = "bucket_lister"
  title       = "Bucket Lister"
  description = "Grants permission to list Cloud Storage buckets"
  permissions = [
    "storage.buckets.list",
    "storage.buckets.get"
  ]
  project = var.project_id
}

resource "google_project_iam_custom_role" "vm_deleter" {
  role_id     = "vm_deleter"
  title       = "VM Deleter"
  description = "Grants permission to delete VMs"
  permissions = [
    "compute.instances.delete"
  ]
  project = var.project_id
}

resource "google_project_iam_custom_role" "image_creator" {
  role_id     = "image_creator"
  title       = "Image Creator"
  description = "Grants permission to create images, instance templates, and disks, and start/stop instances for image creation"
  permissions = [
    "compute.projects.get",
    "compute.instances.start",
    "compute.instances.stop",
    "compute.instances.get",
    "compute.instances.getGuestAttributes",
    "compute.instances.setMetadata",
    "compute.disks.useReadOnly",
    "compute.disks.use",
    "compute.disks.get",
    "compute.images.create",
    "compute.images.get",
    "compute.globalOperations.get"
  ]
  project = var.project_id
}

resource "google_project_iam_custom_role" "template_creator" {
  role_id     = "template_creator"
  title       = "Template Creator"
  description = "Grants permission to create instance templates"
  permissions = [
    "compute.images.get",
    "compute.instanceTemplates.create",
    "compute.instanceTemplates.get",
    "compute.networks.get",
    "compute.networks.use",
    "compute.disks.use",
    "compute.zoneOperations.get"
  ]
  project = var.project_id
}

resource "google_storage_bucket_iam_member" "pipeline_zen_jobs_results" {
  for_each = local.buckets

  bucket = google_storage_bucket.pipeline_zen_results[each.key].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipeline_zen_jobs.email}"
}

resource "google_storage_bucket_iam_member" "pipeline_zen_jobs_datasets" {
  for_each = google_storage_bucket.pipeline_zen_datasets

  bucket = each.value.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipeline_zen_jobs.email}"
}

resource "google_project_iam_member" "pipeline_zen_jobs_project" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
    "roles/secretmanager.viewer",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/compute.viewer"
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.pipeline_zen_jobs.email}"
}

resource "google_project_iam_member" "pipeline_zen_jobs_custom_roles" {
  for_each = toset([
    google_project_iam_custom_role.bucket_lister.role_id,
    google_project_iam_custom_role.vm_deleter.role_id
    ])

  project = var.project_id
  role    = "projects/${var.project_id}/roles/${each.key}"
  member  = "serviceAccount:${google_service_account.pipeline_zen_jobs.email}"
}