#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

DOCKER_BINARY=${DOCKER_BINARY:-docker}
DOCKER_OPTS=${DOCKER_OPTS:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}

set -euo pipefail

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

trap 'error_handler ${LINENO}' ERR

if [ $# -lt 2 ]; then
  echo "Usage: $0 <container name> <build file path>"
  exit 1
fi

docker_image="$1"
host_build_dir="$2"
container_build_dir="/app/builds"

if command -v selinuxenabled &> /dev/null && selinuxenabled; then
  # Disable SELinux labels
  DOCKER_OPTS="$DOCKER_OPTS --security-opt label=disable"
fi

mounts=""
if [ -n "$LLAMA_STACK_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):/app/llama-stack-source"
fi

set -x
$DOCKER_BINARY run $DOCKER_OPTS -it \
  --entrypoint "/usr/local/bin/llama" \
  -v $host_build_dir:$container_build_dir \
  $mounts \
  $docker_image \
  stack configure ./llamastack-build.yaml --output-dir $container_build_dir
