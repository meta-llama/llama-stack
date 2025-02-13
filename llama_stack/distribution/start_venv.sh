#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m' # No Color

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

trap 'error_handler ${LINENO}' ERR

if [ $# -lt 3 ]; then
  echo "Usage: $0 <venv_path> <yaml_config> <port> <script_args...>"
  exit 1
fi

venv_path="$1"
shift

yaml_config="$1"
shift

port="$1"
shift

# Initialize env_vars as an empty array
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

# Activate virtual environment
if [ ! -d "$venv_path" ]; then
  echo -e "${RED}Error: Virtual environment not found at $venv_path${NC}" >&2
  exit 1
fi

source "$venv_path/bin/activate"

set -x
python -m llama_stack.distribution.server.server \
  --yaml-config "$yaml_config" \
  --port "$port" \
  $env_vars \
  $other_args
