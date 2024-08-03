#!/bin/bash

# This script creates Google Cloud Pub/Sub subscriptions for a list of predefined clusters.
#
# It performs the following actions:
# 1. Defines a list of cluster names.
# 2. For each cluster, it constructs a subscription name in the format:
#    projects/[PROJECT_ID]/subscriptions/pipeline-zen-jobs-start-[CLUSTER_NAME]
# 3. Checks if the subscription already exists.
# 4. If the subscription doesn't exist, it creates a new subscription with the following properties:
#    - Attached to the specified topic
#    - 60-second acknowledgement deadline
#    - 4-day message retention
#    - No expiration
#    - Message filter based on the cluster name
#    - Exactly-once delivery enabled
# 5. Skips creation if the subscription already exists.
#
# Usage: ./create-subscriptions.sh

PROJECT_ID="neat-airport-407301"
TOPIC="projects/neat-airport-407301/topics/pipeline-zen-jobs-start"

CLUSTERS=(
    "1xa100-40gb" "2xa100-40gb" "4xa100-40gb" "8xa100-40gb"
    "1xa100-80gb" "2xa100-80gb" "4xa100-80gb" "8xa100-80gb"
    "8xh100-80gb" "1xv100" "4xv100" "8xv100"
)

subscription_exists() {
    gcloud pubsub subscriptions describe "$1" --project="$PROJECT_ID" &> /dev/null
}

create_subscription() {
    local subscription_name="$1"
    local cluster="$2"
    
    gcloud pubsub subscriptions create "$subscription_name" \
        --topic="$TOPIC" \
        --ack-deadline=60 \
        --message-retention-duration=4d \
        --expiration-period=never \
        --message-filter="attributes.cluster = \"$cluster\"" \
        --enable-exactly-once-delivery \
        --project="$PROJECT_ID"
    
    if [ $? -eq 0 ]; then
        echo "Successfully created subscription $subscription_name"
    else
        echo "Error creating subscription $subscription_name"
    fi
}

for cluster in "${CLUSTERS[@]}"; do
    subscription_name="projects/$PROJECT_ID/subscriptions/pipeline-zen-jobs-start-$cluster"
    if subscription_exists "$subscription_name"; then
        echo "Subscription $subscription_name already exists. Skipping."
    else
        create_subscription "$subscription_name" "$cluster"
    fi
done