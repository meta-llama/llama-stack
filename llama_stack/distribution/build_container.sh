#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}

TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
PYPI_VERSION=${PYPI_VERSION:-}
BUILD_PLATFORM=${BUILD_PLATFORM:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}

# mounting is not supported by docker buildx, so we use COPY instead
USE_COPY_NOT_MOUNT=${USE_COPY_NOT_MOUNT:-}

if [ "$#" -lt 4 ]; then
  # This only works for templates
  echo "Usage: $0 <template_or_config> <image_name> <container_base> <pip_dependencies> [<special_pip_deps>]" >&2
  exit 1
fi

set -euo pipefail

template_or_config="$1"
shift
image_name="$1"
shift
container_base="$1"
shift
pip_dependencies="$1"
shift
special_pip_deps="${1:-}"


# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

CONTAINER_BINARY=${CONTAINER_BINARY:-docker}
CONTAINER_OPTS=${CONTAINER_OPTS:---progress=plain}

TEMP_DIR=$(mktemp -d)

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

add_to_container() {
  output_file="$TEMP_DIR/Containerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    # If stdin is not a terminal, read from it (heredoc)
    cat >>"$output_file"
  fi
}

# Check if container command is available
if ! is_command_available $CONTAINER_BINARY; then
  printf "${RED}Error: ${CONTAINER_BINARY} command not found. Is ${CONTAINER_BINARY} installed and in your PATH?${NC}" >&2
  exit 1
fi

# Update and install UBI9 components if UBI9 base image is used
if [[ $container_base == *"registry.access.redhat.com/ubi9"* ]]; then
  add_to_container << EOF
FROM $container_base
WORKDIR /app

RUN dnf -y update && dnf install -y iputils net-tools wget \
    vim-minimal python3.11 python3.11-pip python3.11-wheel \
    python3.11-setuptools && ln -s /bin/pip3.11 /bin/pip && ln -s /bin/python3.11 /bin/python && dnf clean all

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
       gcc \
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
client_mount="/app/llama-stack-client-source"

install_local_package() {
  local dir="$1"
  local mount_point="$2"
  local name="$3"

  if [ ! -d "$dir" ]; then
    echo "${RED}Warning: $name is set but directory does not exist: $dir${NC}" >&2
    exit 1
  fi

  if [ "$USE_COPY_NOT_MOUNT" = "true" ]; then
    add_to_container << EOF
COPY $dir $mount_point
EOF
  fi
  add_to_container << EOF
RUN uv pip install --no-cache -e $mount_point
EOF
}


if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  install_local_package "$LLAMA_STACK_CLIENT_DIR" "$client_mount" "LLAMA_STACK_CLIENT_DIR"
fi

if [ -n "$LLAMA_STACK_DIR" ]; then
  install_local_package "$LLAMA_STACK_DIR" "$stack_mount" "LLAMA_STACK_DIR"
else
  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    add_to_container << EOF
RUN uv pip install fastapi libcst
EOF
    add_to_container << EOF
RUN uv pip install --no-cache --extra-index-url https://test.pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  llama-stack==$TEST_PYPI_VERSION

EOF
  else
    if [ -n "$PYPI_VERSION" ]; then
      SPEC_VERSION="llama-stack==${PYPI_VERSION}"
    else
      SPEC_VERSION="llama-stack"
    fi
    add_to_container << EOF
RUN uv pip install --no-cache $SPEC_VERSION
EOF
  fi
fi

# remove uv after installation
  add_to_container << EOF
RUN pip uninstall -y uv
EOF

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

# Add other require item commands genearic to all containers
add_to_container << EOF

# Allows running as non-root user
RUN mkdir -p /.llama /.cache

RUN chmod -R g+rw /app /.llama /.cache
EOF

printf "Containerfile created successfully in %s/Containerfile\n\n" "$TEMP_DIR"
cat "$TEMP_DIR"/Containerfile
printf "\n"

# Start building the CLI arguments
CLI_ARGS=()

# Read CONTAINER_OPTS and put it in an array
read -ra CLI_ARGS <<< "$CONTAINER_OPTS"

if [ "$USE_COPY_NOT_MOUNT" != "true" ]; then
  if [ -n "$LLAMA_STACK_DIR" ]; then
    CLI_ARGS+=("-v" "$(readlink -f "$LLAMA_STACK_DIR"):$stack_mount")
  fi
  if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
    CLI_ARGS+=("-v" "$(readlink -f "$LLAMA_STACK_CLIENT_DIR"):$client_mount")
  fi
fi

if is_command_available selinuxenabled && selinuxenabled; then
  # Disable SELinux labels -- we don't want to relabel the llama-stack source dir
  CLI_ARGS+=("--security-opt" "label=disable")
fi

# Set version tag based on PyPI version
if [ -n "$PYPI_VERSION" ]; then
  version_tag="$PYPI_VERSION"
elif [ -n "$TEST_PYPI_VERSION" ]; then
  version_tag="test-$TEST_PYPI_VERSION"
elif [[ -n "$LLAMA_STACK_DIR" || -n "$LLAMA_STACK_CLIENT_DIR" ]]; then
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
  CLI_ARGS+=("--platform" "$BUILD_PLATFORM")
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  CLI_ARGS+=("--platform" "linux/arm64")
elif [ "$ARCH" = "x86_64" ]; then
  CLI_ARGS+=("--platform" "linux/amd64")
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

echo "PWD: $(pwd)"
echo "Containerfile: $TEMP_DIR/Containerfile"
set -x

$CONTAINER_BINARY build \
  "${CLI_ARGS[@]}" \
  -t "$image_tag" \
  -f "$TEMP_DIR/Containerfile" \
  "."

# clean up tmp/configs
set +x

echo "Success!"
