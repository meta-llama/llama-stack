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
GREEN='\033[0;32m'
NC='\033[0m' # No Color

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

trap 'error_handler ${LINENO}' ERR

if [ $# -lt 3 ]; then
  echo "Usage: $0 <env_type> <env_path_or_name> <port> [--config <yaml_config>] [--env KEY=VALUE]..."
  exit 1
fi

env_type="$1"
shift

env_path_or_name="$1"
container_image="localhost/$env_path_or_name"
shift

port="$1"
shift

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# Initialize variables
yaml_config=""
env_vars=""
other_args=""

# Process remaining arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ -n "$2" ]]; then
        yaml_config="$2"
        shift 2
      else
        echo -e "${RED}Error: $1 requires a CONFIG argument${NC}" >&2
        exit 1
      fi
      ;;
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

# Check if yaml_config is required based on env_type
if [[ "$env_type" == "venv" || "$env_type" == "conda" ]] && [ -z "$yaml_config" ]; then
  echo -e "${RED}Error: --config is required for venv and conda environments${NC}" >&2
  exit 1
fi

PYTHON_BINARY="python"
case "$env_type" in
  "venv")
    if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" == "$env_path_or_name" ]; then
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

    if [ -n "$yaml_config" ]; then
        yaml_config_arg="--config $yaml_config"
    else
        yaml_config_arg=""
    fi

    $PYTHON_BINARY -m llama_stack.distribution.server.server \
    $yaml_config_arg \
    --port "$port" \
    $env_vars \
    $other_args
elif [[ "$env_type" == "container" ]]; then
    # Determine the internal container address based on container runtime
    if [ "$CONTAINER_BINARY" = "docker" ]; then
        internal_host="host.docker.internal"
    elif [ "$CONTAINER_BINARY" = "podman" ]; then
        internal_host="host.containers.internal"
    else
        internal_host="localhost"
    fi
    echo -e "${RED}Warning: Llama Stack no longer supports running Container.${NC}"
    echo -e "Please use one of the following alternatives:"
    echo -e "1. Use venv or conda environments"
    echo -e "2. Run the container directly with Docker/Podman"
    echo -e "\nExample $CONTAINER_BINARY command for ollama distribution:"
    echo -e "$CONTAINER_BINARY run \\"
    echo -e "  -it \\"
    echo -e "  --network host \\"
    echo -e "  -p $port:$port \\"
    echo -e "  -v <path_to_yaml_config>:/app/run.yaml \\"
    echo -e "  --entrypoint python \\"
    echo -e "  localhost/distribution-ollama:<version> \\"
    echo -e "  -m llama_stack.distribution.server.server \\"
    echo -e "  --config /app/run.yaml \\"
    echo -e "  --env INFERENCE_MODEL=\"llama3.2:3b\" \\"
    echo -e "  --env LLAMA_STACK_PORT=<port> \\"
    echo -e "  --env OLLAMA_URL=\"http://$internal_host:11434\""
    echo -e "\nExiting..."
    exit 1
fi
