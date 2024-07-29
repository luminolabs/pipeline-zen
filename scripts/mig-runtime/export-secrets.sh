#!/bin/bash

# Export predefined secrets in a project as environment variables to be used by the run_remote.sh script

set -e  # Exit immediately if a command fails

# Load utility functions
source ./scripts/utils.sh

# Define the list of secrets to be imported
SECRETS_TO_IMPORT=(
  "huggingface_token"
  # Add more secret names as needed
)

# Function to access a secret value
access_secret_value() {
  local secret_name=$1
  gcloud secrets versions access latest --secret="$secret_name" --project="$PROJECT_ID"
}

# Iterate through the predefined list of secrets and export their values as environment variables
echo "Exporting predefined secrets as environment variables..."
for secret_name in "${SECRETS_TO_IMPORT[@]}"; do
  if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &>/dev/null; then
    secret_value=$(access_secret_value "$secret_name")
    env_var_name="PZ_$(echo "$secret_name" | tr '[:lower:]' '[:upper:]')"
    export $env_var_name=$secret_value
    echo "...exported $env_var_name"
  else
    echo "Warning: Secret $secret_name not found in project $PROJECT_ID"
  fi
done
echo "Predefined secrets exported successfully."