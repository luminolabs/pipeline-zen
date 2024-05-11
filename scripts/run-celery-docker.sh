#!/bin/bash

image_remote=us-central1-docker.pkg.dev/neat-airport-407301/lum-docker-images/train_evaluate-workflow:latest
image_local=train_evaluate-workflow:local

env=""
image_use=$image_local
if [[ "$PZ_ENV" != "" ]]; then
  env=$PZ_ENV
  image_use=$image_remote
fi

if [[ "$image_use" == "$image_local" ]]; then
  docker build -f celery.Dockerfile -t $image_use .
else
  docker pull $image_use
fi

gpus="--gpus all"
if [[ "$OSTYPE" == "darwin"* ]]; then
  # There's no implementation on OSX to allow using the GPU with Docker;
  # this means that MPS will not be used
  # when running ML workflows on Docker under OSX (ie. the Mac GPU won't be used)
  gpus=""
fi

docker run $gpus \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV=$env \
$image_use "${@}"