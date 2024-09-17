#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

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

set -x
podman run -it \
  -v $host_build_dir:$container_build_dir \
  $docker_image \
  llama stack configure ./llamastack-build.yaml --output-dir $container_build_dir
