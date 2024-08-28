#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <image_name> <base_image> <pip_dependencies>
  echo "Example: $0 my-fastapi-app python:3.9-slim 'fastapi uvicorn'
  exit 1
fi

IMAGE_NAME=$1
BASE_IMAGE=$2
PIP_DEPENDENCIES=$3

set -euo pipefail

PORT=8001

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
SOURCE_DIR=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
echo $SOURCE_DIR

TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

cat <<EOF >"$TEMP_DIR/Dockerfile"
FROM $BASE_IMAGE
WORKDIR /app
COPY llama-stack/llama_toolchain /app/llama_toolchain
COPY llama-models/models /app/llama_models

RUN pip install $PIP_DEPENDENCIES

EXPOSE $PORT
ENTRYPOINT ["python3", "-m", "llama_toolchain.distribution.server", "--port", "$PORT"]

EOF

echo "Dockerfile created successfully in $TEMP_DIR/Dockerfile"

podman build -t $IMAGE_NAME -f "$TEMP_DIR/Dockerfile" "$SOURCE_DIR"

echo "Podman image '$IMAGE_NAME' built successfully."
echo "You can run it with: podman run -p 8000:8000 $IMAGE_NAME"

rm -rf "$TEMP_DIR"
