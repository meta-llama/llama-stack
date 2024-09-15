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
build_file_path="$2"

podman run -it $docker_image cat build.yaml >> ./$build_file_path
