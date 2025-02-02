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

# mounting is not supported by docker buildx, so we use COPY instead
USE_COPY_NOT_MOUNT=${USE_COPY_NOT_MOUNT:-}

if [ "$#" -lt 6 ]; then
  # This only works for templates
  echo "Usage: $0 <template_or_config> <image_name> <container_base> <build_file_path> <host_build_dir> <pip_dependencies> [<special_pip_deps>]" >&2
  exit 1
fi

set -euo pipefail

template_or_config="$1"
image_name="$2"
container_base="$3"
build_file_path="$4"
host_build_dir="$5"
pip_dependencies="$6"
special_pip_deps="$7"


# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

CONTAINER_BINARY=${CONTAINER_BINARY:-docker}
CONTAINER_OPTS=${CONTAINER_OPTS:-}

TEMP_DIR=$(mktemp -d)

add_to_container() {
  local input
  output_file="$TEMP_DIR/Containerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    # If stdin is not a terminal, read from it (heredoc)
    cat >>"$output_file"
  fi
}

# Update and install UBI9 components if UBI9 base image is used
if [[ $container_base == *"registry.access.redhat.com/ubi9"* ]]; then
  add_to_container << EOF
FROM $container_base
WORKDIR /app

RUN microdnf -y update && microdnf install -y iputils net-tools wget \
    vim-minimal python3.11 python3.11-pip python3.11-wheel \
    python3.11-setuptools && ln -s /bin/pip3.11 /bin/pip && ln -s /bin/python3.11 /bin/python && microdnf clean all

ENV UV_SYSTEM_PYTHON=1
RUN pip install uv
EOF
else
  add_to_container << EOF
FROM $container_base
WORKDIR /app

RUN apt-get update && apt-get install -y \
       iputils-ping net-tools iproute2 dnsutils telnet \
       curl wget telnet \
       procps psmisc lsof \
       traceroute \
       bubblewrap \
       && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1
RUN pip install uv
EOF
fi

# Add pip dependencies first since llama-stack is what will change most often
# so we can reuse layers.
if [ -n "$pip_dependencies" ]; then
  add_to_container << EOF
RUN uv pip install --no-cache $pip_dependencies
EOF
fi

if [ -n "$special_pip_deps" ]; then
  IFS='#' read -ra parts <<<"$special_pip_deps"
  for part in "${parts[@]}"; do
    add_to_container <<EOF
RUN uv pip install --no-cache $part
EOF
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

  if [ "$USE_COPY_NOT_MOUNT" = "true" ]; then
    add_to_container << EOF
COPY $LLAMA_STACK_DIR $stack_mount
EOF
  fi

  add_to_container << EOF
RUN uv pip install --no-cache -e $stack_mount
EOF
else
  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    add_to_container << EOF
RUN uv pip install fastapi libcst
EOF
    add_to_container << EOF
RUN uv pip install --no-cache --extra-index-url https://test.pypi.org/simple/ \
  llama-models==$TEST_PYPI_VERSION llama-stack-client==$TEST_PYPI_VERSION llama-stack==$TEST_PYPI_VERSION

EOF
  else
    if [ -n "$PYPI_VERSION" ]; then
      SPEC_VERSION="llama-stack==${PYPI_VERSION} llama-models==${PYPI_VERSION} llama-stack-client==${PYPI_VERSION}"
    else
      SPEC_VERSION="llama-stack"
    fi
    add_to_container << EOF
RUN uv pip install --no-cache $SPEC_VERSION
EOF
  fi
fi

if [ -n "$LLAMA_MODELS_DIR" ]; then
  if [ ! -d "$LLAMA_MODELS_DIR" ]; then
    echo "${RED}Warning: LLAMA_MODELS_DIR is set but directory does not exist: $LLAMA_MODELS_DIR${NC}" >&2
    exit 1
  fi

  if [ "$USE_COPY_NOT_MOUNT" = "true" ]; then
    add_to_container << EOF
COPY $LLAMA_MODELS_DIR $models_mount
EOF
  fi
  add_to_container << EOF
RUN uv pip uninstall llama-models
RUN uv pip install --no-cache $models_mount
EOF
fi

# if template_or_config ends with .yaml, it is not a template and we should not use the --template flag
if [[ "$template_or_config" != *.yaml ]]; then
  add_to_container << EOF
ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server", "--template", "$template_or_config"]
EOF
else
  add_to_container << EOF
ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server"]
EOF
fi

printf "Containerfile created successfully in $TEMP_DIR/Containerfile\n\n"
cat $TEMP_DIR/Containerfile
printf "\n"

mounts=""
if [ "$USE_COPY_NOT_MOUNT" != "true" ]; then
  if [ -n "$LLAMA_STACK_DIR" ]; then
    mounts="$mounts -v $(readlink -f $LLAMA_STACK_DIR):$stack_mount"
  fi
  if [ -n "$LLAMA_MODELS_DIR" ]; then
    mounts="$mounts -v $(readlink -f $LLAMA_MODELS_DIR):$models_mount"
  fi
fi

if command -v selinuxenabled &>/dev/null && selinuxenabled; then
  # Disable SELinux labels -- we don't want to relabel the llama-stack source dir
  CONTAINER_OPTS="$CONTAINER_OPTS --security-opt label=disable"
fi

# Set version tag based on PyPI version
if [ -n "$PYPI_VERSION" ]; then
  version_tag="$PYPI_VERSION"
elif [ -n "$TEST_PYPI_VERSION" ]; then
  version_tag="test-$TEST_PYPI_VERSION"
elif [[ -n "$LLAMA_STACK_DIR" || -n "$LLAMA_MODELS_DIR" ]]; then
  version_tag="dev"
else
  URL="https://pypi.org/pypi/llama-stack/json"
  version_tag=$(curl -s $URL | jq -r '.info.version')
fi

# Add version tag to image name
image_tag="$image_name:$version_tag"

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

echo "PWD: $(pwd)"
echo "Containerfile: $TEMP_DIR/Containerfile"
set -x
$CONTAINER_BINARY build $CONTAINER_OPTS $PLATFORM -t $image_tag \
  -f "$TEMP_DIR/Containerfile" "." $mounts --progress=plain

# clean up tmp/configs
set +x

echo "Success!"
