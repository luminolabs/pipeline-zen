docker build -f workflows.Dockerfile --build-arg TARGET_WORKFLOW=$1 -t "$1-workflow:local" .

env=""
if [[ "$PZ_ENV" != "" ]]; then
  env=$PZ_ENV
fi

gpus="--gpus all"
if [[ "$OSTYPE" == "darwin"* ]]; then
  # There's no implementation on OSX to allow using the GPU with Docker;
  # this means that MPS will not be used
  # when running ML workflows on Docker under OSX (ie. the Mac GPU won't be used)
  gpus=""
fi

docker run "$gpus" \
-v "$PWD/.cache":/project/.cache \
-v "$PWD/.results":/project/.results \
-v "$PWD/.logs":/project/.logs \
-v "$PWD/.secrets":/project/.secrets \
-e PZ_ENV="$env" \
"$1-workflow:local" "${@:2}"