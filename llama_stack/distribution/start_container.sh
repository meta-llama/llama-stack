#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

DOCKER_BINARY=${DOCKER_BINARY:-docker}
DOCKER_OPTS=${DOCKER_OPTS:-}
LLAMA_CHECKPOINT_DIR=${LLAMA_CHECKPOINT_DIR:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}

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

build_name="$1"
docker_image="llamastack-$build_name"
shift

yaml_config="$1"
shift

port="$1"
shift

set -x

if command -v selinuxenabled &> /dev/null && selinuxenabled; then
  # Disable SELinux labels
  DOCKER_OPTS="$DOCKER_OPTS --security-opt label=disable"
fi

mounts=""
if [ -n "$LLAMA_STACK_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):/app/llama-stack-source"
fi
if [ -n "$LLAMA_CHECKPOINT_DIR" ]; then
  mounts="$mounts -v $LLAMA_CHECKPOINT_DIR:/root/.llama"
  DOCKER_OPTS="$DOCKER_OPTS --gpus=all"
fi

$DOCKER_BINARY run $DOCKER_OPTS -it \
  -p $port:$port \
  -v "$yaml_config:/app/config.yaml" \
  $mounts \
  $docker_image \
  python -m llama_stack.distribution.server.server \
  --yaml_config /app/config.yaml \
  --port $port "$@"
