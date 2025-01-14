#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
PYPI_VERSION=${PYPI_VERSION:-}
BUILD_PLATFORM=${BUILD_PLATFORM:-}

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <build_name> <docker_base> <pip_dependencies> [<special_pip_deps>]" >&2
  echo "Example: $0 my-fastapi-app python:3.9-slim 'fastapi uvicorn' " >&2
  exit 1
fi

special_pip_deps="$7"

set -euo pipefail

build_name="$1"
image_name="distribution-$build_name"
docker_base=$2
build_file_path=$3
host_build_dir=$4
pip_dependencies=$5
dockerfile_only=$6

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname $(dirname "$SCRIPT_DIR"))
DOCKER_BINARY=${DOCKER_BINARY:-docker}
DOCKER_OPTS=${DOCKER_OPTS:-}

TEMP_DIR=$(mktemp -d)

add_to_docker() {
  local input
  output_file="$TEMP_DIR/Dockerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    # If stdin is not a terminal, read from it (heredoc)
    cat >>"$output_file"
  fi
}

# Update and install UBI9 components if UBI9 base image is used
if [[ $docker_base == *"registry.access.redhat.com/ubi9"* ]]; then
  add_to_docker <<EOF
FROM $docker_base
WORKDIR /app

RUN microdnf -y update && microdnf install -y iputils net-tools wget \
    vim-minimal python3.11 python3.11-pip python3.11-wheel \
    python3.11-setuptools && ln -s /bin/pip3.11 /bin/pip && ln -s /bin/python3.11 /bin/python && microdnf clean all

EOF
else
  add_to_docker <<EOF
FROM $docker_base
WORKDIR /app

RUN apt-get update && apt-get install -y \
       iputils-ping net-tools iproute2 dnsutils telnet \
       curl wget telnet \
       procps psmisc lsof \
       traceroute \
       bubblewrap \
       && rm -rf /var/lib/apt/lists/*

EOF
fi

# Add pip dependencies first since llama-stack is what will change most often
# so we can reuse layers.
if [ -n "$pip_dependencies" ]; then
  add_to_docker "RUN pip install --no-cache $pip_dependencies"
fi

if [ -n "$special_pip_deps" ]; then
  IFS='#' read -ra parts <<<"$special_pip_deps"
  for part in "${parts[@]}"; do
    add_to_docker "RUN pip install --no-cache $part"
  done
fi

stack_mount="/app/llama-stack-source"
models_mount="/app/llama-models-source"

if [ -n "$LLAMA_STACK_DIR" ]; then
  if [ ! -d "$LLAMA_STACK_DIR" ]; then
    echo "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: $LLAMA_STACK_DIR${NC}" >&2
    exit 1
  fi

  # Install in editable format. We will mount the source code into the container
  # so that changes will be reflected in the container without having to do a
  # rebuild. This is just for development convenience.
  add_to_docker "RUN pip install --no-cache -e $stack_mount"
else
  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    add_to_docker "RUN pip install fastapi libcst"
    add_to_docker <<EOF
RUN pip install --no-cache --extra-index-url https://test.pypi.org/simple/ \
  llama-models==$TEST_PYPI_VERSION llama-stack-client==$TEST_PYPI_VERSION llama-stack==$TEST_PYPI_VERSION
EOF
  else
    if [ -n "$PYPI_VERSION" ]; then
      SPEC_VERSION="llama-stack==${PYPI_VERSION} llama-models==${PYPI_VERSION} llama-stack-client==${PYPI_VERSION}"
    else
      SPEC_VERSION="llama-stack"
    fi
    add_to_docker "RUN pip install --no-cache $SPEC_VERSION"
  fi
fi

if [ -n "$LLAMA_MODELS_DIR" ]; then
  if [ ! -d "$LLAMA_MODELS_DIR" ]; then
    echo "${RED}Warning: LLAMA_MODELS_DIR is set but directory does not exist: $LLAMA_MODELS_DIR${NC}" >&2
    exit 1
  fi

  add_to_docker <<EOF
RUN pip uninstall -y llama-models
RUN pip install --no-cache $models_mount

EOF
fi

add_to_docker <<EOF

# This would be good in production but for debugging flexibility lets not add it right now
# We need a more solid production ready entrypoint.sh anyway
#
ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server", "--template", "$build_name"]

EOF

printf "Dockerfile created successfully in $TEMP_DIR/Dockerfile\n\n"
cat $TEMP_DIR/Dockerfile
printf "\n"

mounts=""
if [ -n "$LLAMA_STACK_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):$stack_mount"
fi
if [ -n "$LLAMA_MODELS_DIR" ]; then
  mounts="$mounts -v $(readlink -f $LLAMA_MODELS_DIR):$models_mount"
fi

if command -v selinuxenabled &>/dev/null && selinuxenabled; then
  # Disable SELinux labels -- we don't want to relabel the llama-stack source dir
  DOCKER_OPTS="$DOCKER_OPTS --security-opt label=disable"
fi

# Set version tag based on PyPI version
if [ -n "$TEST_PYPI_VERSION" ]; then
  version_tag="test-$TEST_PYPI_VERSION"
elif [[ -n "$LLAMA_STACK_DIR" || -n "$LLAMA_MODELS_DIR" ]]; then
  version_tag="dev"
else
  URL="https://pypi.org/pypi/llama-stack/json"
  version_tag=$(curl -s $URL | jq -r '.info.version')
fi

# Add version tag to image name
image_tag="$image_name:$version_tag"

if [ "$dockerfile_only" = "True" ]; then
  echo "Skipping docker build as dockerfile_only=True"
  dockerfile_name="Dockerfile.${image_tag}"
  cp "$TEMP_DIR/Dockerfile" "./${dockerfile_name}"
  printf "Dockerfile saved to ./${dockerfile_name}\n\n"
  exit 0
fi

# Detect platform architecture
ARCH=$(uname -m)
if [ -n "$BUILD_PLATFORM" ]; then
  PLATFORM="--platform $BUILD_PLATFORM"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  PLATFORM="--platform linux/arm64"
elif [ "$ARCH" = "x86_64" ]; then
  PLATFORM="--platform linux/amd64"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

set -x
$DOCKER_BINARY build $DOCKER_OPTS $PLATFORM -t $image_tag -f "$TEMP_DIR/Dockerfile" "$REPO_DIR" $mounts

# clean up tmp/configs
set +x

echo "Success!"
