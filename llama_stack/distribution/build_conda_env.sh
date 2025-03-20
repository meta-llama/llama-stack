#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}

if [ -n "$LLAMA_STACK_DIR" ]; then
  echo "Using llama-stack-dir=$LLAMA_STACK_DIR"
fi
if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  echo "Using llama-stack-client-dir=$LLAMA_STACK_CLIENT_DIR"
fi

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <distribution_type> <conda_env_name> <build_file_path> <pip_dependencies> [<special_pip_deps>]" >&2
  echo "Example: $0 <distribution_type> my-conda-env ./my-stack-build.yaml 'numpy pandas scipy'" >&2
  exit 1
fi

special_pip_deps="$4"

set -euo pipefail

env_name="$1"
build_file_path="$2"
pip_dependencies="$3"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# this is set if we actually create a new conda in which case we need to clean up
ENVNAME=""

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

ensure_conda_env_python310() {
  local env_name="$1"
  local pip_dependencies="$2"
  local special_pip_deps="$3"
  local python_version="3.10"

  # Check if conda command is available
  if ! is_command_available conda; then
    printf "${RED}Error: conda command not found. Is Conda installed and in your PATH?${NC}" >&2
    exit 1
  fi

  # Check if the environment exists
  if conda env list | grep -q "^${env_name} "; then
    printf "Conda environment '${env_name}' exists. Checking Python version...\n"

    # Check Python version in the environment
    current_version=$(conda run -n "${env_name}" python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

    if [ "$current_version" = "$python_version" ]; then
      printf "Environment '${env_name}' already has Python ${python_version}. No action needed.\n"
    else
      printf "Updating environment '${env_name}' to Python ${python_version}...\n"
      conda install -n "${env_name}" python="${python_version}" -y
    fi
  else
    printf "Conda environment '${env_name}' does not exist. Creating with Python ${python_version}...\n"
    conda create -n "${env_name}" python="${python_version}" -y

    ENVNAME="${env_name}"
    # setup_cleanup_handlers
  fi

  eval "$(conda shell.bash hook)"
  conda deactivate && conda activate "${env_name}"

  $CONDA_PREFIX/bin/pip install uv

  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    uv pip install fastapi libcst
    uv pip install --extra-index-url https://test.pypi.org/simple/ \
      llama-stack==$TEST_PYPI_VERSION \
      $pip_dependencies
    if [ -n "$special_pip_deps" ]; then
      IFS='#' read -ra parts <<<"$special_pip_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        uv pip install $part
      done
    fi
  else
    # Re-installing llama-stack in the new conda environment
    if [ -n "$LLAMA_STACK_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: $LLAMA_STACK_DIR${NC}\n" >&2
        exit 1
      fi

      printf "Installing from LLAMA_STACK_DIR: $LLAMA_STACK_DIR\n"
      uv pip install --no-cache-dir -e "$LLAMA_STACK_DIR"
    else
      PYPI_VERSION="${PYPI_VERSION:-}"
      if [ -n "$PYPI_VERSION" ]; then
        SPEC_VERSION="llama-stack==${PYPI_VERSION}"
      else
        SPEC_VERSION="llama-stack"
      fi
      uv pip install --no-cache-dir $SPEC_VERSION
    fi

    if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_CLIENT_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_CLIENT_DIR is set but directory does not exist: $LLAMA_STACK_CLIENT_DIR${NC}\n" >&2
        exit 1
      fi

      printf "Installing from LLAMA_STACK_CLIENT_DIR: $LLAMA_STACK_CLIENT_DIR\n"
      uv pip install --no-cache-dir -e "$LLAMA_STACK_CLIENT_DIR"
    fi

    # Install pip dependencies
    printf "Installing pip dependencies\n"
    uv pip install $pip_dependencies
    if [ -n "$special_pip_deps" ]; then
      IFS='#' read -ra parts <<<"$special_pip_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        uv pip install $part
      done
    fi
  fi

  mv $build_file_path $CONDA_PREFIX/llamastack-build.yaml
  echo "Build spec configuration saved at $CONDA_PREFIX/llamastack-build.yaml"
}

ensure_conda_env_python310 "$env_name" "$pip_dependencies" "$special_pip_deps"
