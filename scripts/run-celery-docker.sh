#!/bin/bash

VERSION=$(cat VERSION)
image_remote=us-central1-docker.pkg.dev/neat-airport-407301/lum-docker-images/celery-workflow:$VERSION
image_local=celery-workflow:local

env="local"
image_use=$image_local
if [[ "$PZ_ENV" != "local" && "$PZ_ENV" != "" ]]; then
  env=$PZ_ENV
  image_use=$image_remote
fi

if [[ "$image_use" == "$image_local" ]]; then
  docker build -f celery.Dockerfile --build-arg TARGET_WORKFLOW=$1 -t $image_use .
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
$image_use "${@:2}"

echo "Celery workflow finished!"

# If we're not on a local env, let's delete the VM that run this job
# TODO: Make this optional
# see: https://linear.app/luminoai/issue/LUM-180/add-options-to-run-remotepy
if [[ "$env" != "local" ]]; then
  echo "Deleting VM..."
  # Run `delete_vm.py` in the background to
  # allow client to disconnect gracefully while
  # the VM is shutting down
  # The following snippet sends script output
  # to `/dev/null`, puts process in background, and
  # disowns process from shell
  # We disown this process so that when the client
  # disconnects, the process can continue to run
  cmd="python ./scripts/delete_vm.py --job_id $(cat ./.results/.started)"
  eval "${cmd}" &>/dev/null & disown;
fi
