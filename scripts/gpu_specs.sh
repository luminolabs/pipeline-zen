#!/bin/bash

# Pulls nvidia gpu specs

gpu_spec=$(nvidia-smi -x -q | yq -p=xml -o=json)
model=$(echo $gpu_spec | jq '.nvidia_smi_log.gpu.product_name')
memory=$(echo $gpu_spec | jq '.nvidia_smi_log.gpu.fb_memory_usage.total')
pwr_limit=$(echo $gpu_spec | jq '.nvidia_smi_log.gpu.gpu_power_readings.default_power_limit')
echo "{\"model\": \"$model\", \"memory\": \"$memory\", \"pwr_limit\": \"$pwr_limit\"}" | sed -r 's/""/"/g'