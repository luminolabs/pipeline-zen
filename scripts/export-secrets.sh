#!/bin/bash

# Export all secrets in a project as environment variables to be used by the run_remote.sh script

# Set the project ID
PROJECT_ID="neat-airport-407301"

# List all secrets in the specified project and get their names
SECRET_NAMES=$(gcloud secrets list --project="$PROJECT_ID" --format="value(name)")

# Function to access a secret value
access_secret_value() {
  local secret_name=$1
  gcloud secrets versions access latest --secret="$secret_name" --project="$PROJECT_ID"
}

# Iterate through each secret name and export its value as an environment variable
for secret_name in $SECRET_NAMES; do
  secret_value=$(access_secret_value "$secret_name")
  env_var_name="PZ_$(echo "$secret_name" | tr '[:lower:]' '[:upper:]')"
  export $env_var_name=$secret_value
  echo "Exported $env_var_name"
done