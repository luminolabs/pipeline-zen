resource "google_secret_manager_secret" "pipeline_zen_jobs_config" {
  secret_id = "pipeline-zen-jobs-config"
  replication {
      auto {}
  }
}

resource "google_secret_manager_secret_version" "config" {
  secret = google_secret_manager_secret.pipeline_zen_jobs_config.id
  secret_data = file("${path.module}/${var.environment}-config.env")
}