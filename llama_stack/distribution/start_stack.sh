#!/usr/bin/env bash

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
VIRTUAL_ENV=${VIRTUAL_ENV:-}

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m' # No Color

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

trap 'error_handler ${LINENO}' ERR

if [ $# -lt 3 ]; then
  echo "Usage: $0 <env_type> <env_path_or_name> <yaml_config> <port> <script_args...>"
  exit 1
fi

env_type="$1"
shift

env_path_or_name="$1"
container_image="localhost/$env_path_or_name"
shift

yaml_config="$1"
shift

port="$1"
shift

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# Initialize env_vars as an string
env_vars=""
other_args=""
# Process environment variables from --env arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  --env)

    if [[ -n "$2" ]]; then
      env_vars="$env_vars --env $2"
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
PYTHON_BINARY="python"
case "$env_type" in
  "venv")
    if [ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" == "$env_path_or_name" ]; then
        echo -e "${GREEN}Virtual environment already activated${NC}" >&2
    else
        # Activate virtual environment
        if [ ! -d "$env_path_or_name" ]; then
            echo -e "${RED}Error: Virtual environment not found at $env_path_or_name${NC}" >&2
            exit 1
        fi

        if [ ! -f "$env_path_or_name/bin/activate" ]; then
            echo -e "${RED}Error: Virtual environment activate binary not found at $env_path_or_name/bin/activate" >&2
            exit 1
        fi

        source "$env_path_or_name/bin/activate"
    fi
    ;;
  "conda")
    if ! is_command_available conda; then
        echo -e "${RED}Error: conda not found" >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda deactivate && conda activate "$env_path_or_name"
    PYTHON_BINARY="$CONDA_PREFIX/bin/python"
    ;;
  *)
esac

if [[ "$env_type" == "venv" || "$env_type" == "conda" ]]; then
    set -x

    $PYTHON_BINARY -m llama_stack.distribution.server.server \
    --yaml-config "$yaml_config" \
    --port "$port" \
    $env_vars \
    $other_args
elif [[ "$env_type" == "container" ]]; then
    set -x

    # Check if container command is available
    if ! is_command_available $CONTAINER_BINARY; then
      printf "${RED}Error: ${CONTAINER_BINARY} command not found. Is ${CONTAINER_BINARY} installed and in your PATH?${NC}" >&2
      exit 1
    fi

    if is_command_available selinuxenabled &> /dev/null && selinuxenabled; then
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
        if ! is_command_available jq; then
            echo -e "${RED}Error: jq not found" >&2
            exit 1
        fi
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
fi
