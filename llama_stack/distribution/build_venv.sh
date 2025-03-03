#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# TODO: combine this with build_conda_env.sh since it is almost identical
# the only difference is that we don't do any conda-specific setup

LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}
UV_SYSTEM_PYTHON=${UV_SYSTEM_PYTHON:-}
VIRTUAL_ENV=${VIRTUAL_ENV:-}

if [ -n "$LLAMA_STACK_DIR" ]; then
  echo "Using llama-stack-dir=$LLAMA_STACK_DIR"
fi
if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  echo "Using llama-stack-client-dir=$LLAMA_STACK_CLIENT_DIR"
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <env_name> <pip_dependencies> [<special_pip_deps>]" >&2
  echo "Example: $0 mybuild ./my-stack-build.yaml 'numpy pandas scipy'" >&2
  exit 1
fi

special_pip_deps="$3"

set -euo pipefail

env_name="$1"
pip_dependencies="$2"

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

# this is set if we actually create a new conda in which case we need to clean up
ENVNAME=""

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# pre-run checks to make sure we can proceed with the installation
pre_run_checks() {
  local env_name="$1"

  if ! is_command_available uv; then
    echo "uv is not installed, trying to install it."
    if ! is_command_available pip; then
      echo "pip is not installed, cannot automatically install 'uv'."
      echo "Follow this link to install it:"
      echo "https://docs.astral.sh/uv/getting-started/installation/"
      exit 1
    else
      pip install uv
    fi
  fi

  # checking if an environment with the same name already exists
  if [ -d "$env_name" ]; then
    echo "Environment '$env_name' already exists, re-using it."
  fi
}

run() {
  local env_name="$1"
  local pip_dependencies="$2"
  local special_pip_deps="$3"

  if [ -n "$UV_SYSTEM_PYTHON" ] || [ "$env_name" == "__system__" ]; then
    echo "Installing dependencies in system Python environment"
    # if env == __system__, ensure we set UV_SYSTEM_PYTHON
    export UV_SYSTEM_PYTHON=1
  elif [ "$VIRTUAL_ENV" == "$env_name" ]; then
    echo "Virtual environment $env_name is already active"
  else
    echo "Using virtual environment $env_name"
    uv venv "$env_name"
    # shellcheck source=/dev/null
    source "$env_name/bin/activate"
  fi

  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    uv pip install fastapi libcst
    # shellcheck disable=SC2086
    # we are building a command line so word splitting is expected
    uv pip install --extra-index-url https://test.pypi.org/simple/ \
      --index-strategy unsafe-best-match \
      llama-stack=="$TEST_PYPI_VERSION" \
      $pip_dependencies
    if [ -n "$special_pip_deps" ]; then
      IFS='#' read -ra parts <<<"$special_pip_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        # shellcheck disable=SC2086
        # we are building a command line so word splitting is expected
        uv pip install $part
      done
    fi
  else
    # Re-installing llama-stack in the new virtual environment
    if [ -n "$LLAMA_STACK_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: %s${NC}\n" "$LLAMA_STACK_DIR" >&2
        exit 1
      fi

      printf "Installing from LLAMA_STACK_DIR: %s\n"  "$LLAMA_STACK_DIR"
      uv pip install --no-cache-dir -e "$LLAMA_STACK_DIR"
    else
      uv pip install --no-cache-dir llama-stack
    fi

    if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_CLIENT_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_CLIENT_DIR is set but directory does not exist: %s${NC}\n" "$LLAMA_STACK_CLIENT_DIR" >&2
        exit 1
      fi

      printf "Installing from LLAMA_STACK_CLIENT_DIR: %s\n" "$LLAMA_STACK_CLIENT_DIR"
      uv pip install --no-cache-dir -e "$LLAMA_STACK_CLIENT_DIR"
    fi

    # Install pip dependencies
    printf "Installing pip dependencies\n"
    # shellcheck disable=SC2086
    # we are building a command line so word splitting is expected
    uv pip install $pip_dependencies
    if [ -n "$special_pip_deps" ]; then
      IFS='#' read -ra parts <<<"$special_pip_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        # shellcheck disable=SC2086
        # we are building a command line so word splitting is expected
        uv pip install $part
      done
    fi
  fi
}

pre_run_checks "$env_name"
run "$env_name" "$pip_dependencies" "$special_pip_deps"
