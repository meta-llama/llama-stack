#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <image_name> <base_image> <pip_dependencies> <entrypoint_command>"
  echo "Example: $0 my-fastapi-app python:3.9-slim 'fastapi uvicorn' 'python3 -m llama_toolchain.distribution.server --port 8000'"
  exit 1
fi

IMAGE_NAME=$1
BASE_IMAGE=$2
PIP_DEPENDENCIES=$3
ENTRYPOINT_COMMAND=$4

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
SOURCE_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

cat <<EOF >"$TEMP_DIR/Dockerfile"
FROM $BASE_IMAGE
WORKDIR /app
COPY llama_toolchain /app

RUN pip install --no-cache-dir $PIP_DEPENDENCIES

EXPOSE 8000
CMD $ENTRYPOINT_COMMAND
EOF

echo "Dockerfile created successfully in $TEMP_DIR/Dockerfile"

podman build -t $IMAGE_NAME -f "$TEMP_DIR/Dockerfile" "$SOURCE_DIR"

echo "Podman image '$IMAGE_NAME' built successfully."
echo "You can run it with: podman run -p 8000:8000 $IMAGE_NAME"

rm -rf "$TEMP_DIR"
