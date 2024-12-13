resource "google_pubsub_topic" "pipeline_zen_jobs_start" {
  name = "pipeline-zen-jobs-start"
  project = var.project_id
}

resource "google_pubsub_subscription" "pipeline_zen_jobs_start" {
  for_each = toset(concat(tolist(keys(local.machine_types)), ["1xlocal", "2xlocal"]))

  name  = "pipeline-zen-jobs-start-runner-${each.key}"
  topic = google_pubsub_topic.pipeline_zen_jobs_start.name
  project = var.project_id

  message_retention_duration = "345600s"  # 4 days
  retain_acked_messages = false
  ack_deadline_seconds = 60
  expiration_policy {
    ttl = ""  # Never expire
  }
  enable_exactly_once_delivery = true

  filter = "attributes.cluster = \"${each.key}\""
}

resource "google_pubsub_topic" "pipeline_zen_jobs_stop" {
  name = "pipeline-zen-jobs-stop"
  project = var.project_id
}

resource "google_pubsub_subscription" "pipeline_zen_jobs_stop_runner" {
  name  = "pipeline-zen-jobs-stop-runner"
  topic = google_pubsub_topic.pipeline_zen_jobs_stop.name
  project = var.project_id

  message_retention_duration = "345600s"  # 4 days
  retain_acked_messages = false
  ack_deadline_seconds = 60
  expiration_policy {
    ttl = ""  # Never expire
  }
  enable_exactly_once_delivery = true
}

resource "google_pubsub_topic" "pipeline_zen_jobs_heartbeats" {
  for_each = toset(["pipeline-zen-jobs-heartbeats", "pipeline-zen-jobs-heartbeats-local"])

  name = each.key
  project = var.project_id
}

# Create jobs metadata topic
resource "google_pubsub_topic" "pipeline_zen_jobs_meta" {
  for_each = toset(["pipeline-zen-jobs-meta", "pipeline-zen-jobs-meta-local"])

  name = each.key
  project = var.project_id
}