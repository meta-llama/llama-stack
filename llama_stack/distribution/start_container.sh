#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

CONTAINER_BINARY=${CONTAINER_BINARY:-docker}
CONTAINER_OPTS=${CONTAINER_OPTS:-}
LLAMA_CHECKPOINT_DIR=${LLAMA_CHECKPOINT_DIR:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
PYPI_VERSION=${PYPI_VERSION:-}

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m' # No Color

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

trap 'error_handler ${LINENO}' ERR

if [ $# -lt 3 ]; then
  echo "Usage: $0 <build_name> <yaml_config> <port> <other_args...>"
  exit 1
fi

image_name="$1"
container_image="localhost/$image_name"
shift

yaml_config="$1"
shift

port="$1"
shift

# Initialize other_args
other_args=""

# Process environment variables from --env arguments
env_vars=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            echo "env = $2"
            if [[ -n "$2" ]]; then
                env_vars="$env_vars -e $2"
                shift 2
            else
                echo -e "${RED}Error: --env requires a KEY=VALUE argument${NC}" >&2
                exit 1
            fi
            ;;
        *)
            other_args="$other_args $1"
            shift
            ;;
    esac
done

set -x

if command -v selinuxenabled &> /dev/null && selinuxenabled; then
  # Disable SELinux labels
  CONTAINER_OPTS="$CONTAINER_OPTS --security-opt label=disable"
fi

mounts=""
if [ -n "$LLAMA_STACK_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):/app/llama-stack-source"
fi
if [ -n "$LLAMA_CHECKPOINT_DIR" ]; then
  mounts="$mounts -v $LLAMA_CHECKPOINT_DIR:/root/.llama"
  CONTAINER_OPTS="$CONTAINER_OPTS --gpus=all"
fi

if [ -n "$PYPI_VERSION" ]; then
  version_tag="$PYPI_VERSION"
elif [ -n "$LLAMA_STACK_DIR" ]; then
  version_tag="dev"
elif [ -n "$TEST_PYPI_VERSION" ]; then
  version_tag="test-$TEST_PYPI_VERSION"
else
  URL="https://pypi.org/pypi/llama-stack/json"
  version_tag=$(curl -s $URL | jq -r '.info.version')
fi

$CONTAINER_BINARY run $CONTAINER_OPTS -it \
  -p $port:$port \
  $env_vars \
  -v "$yaml_config:/app/config.yaml" \
  $mounts \
  --env LLAMA_STACK_PORT=$port \
  --entrypoint python \
  $container_image:$version_tag \
  -m llama_stack.distribution.server.server \
  --yaml-config /app/config.yaml \
  $other_args
