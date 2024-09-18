#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_MODELS_DIR=${LLAMA_MODELS_DIR:-}
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}

if [ -n "$LLAMA_STACK_DIR" ]; then
  echo "Using llama-stack-dir=$LLAMA_STACK_DIR"
fi
if [ -n "$LLAMA_MODELS_DIR" ]; then
  echo "Using llama-models-dir=$LLAMA_MODELS_DIR"
fi

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <distribution_type> <build_name> <pip_dependencies>" >&2
  echo "Example: $0 <distribution_type> mybuild 'numpy pandas scipy'" >&2
  exit 1
fi

build_name="$1"
env_name="llamastack-$build_name"
pip_dependencies="$2"

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
  local python_version="3.10"

  # Check if conda command is available
  if ! command -v conda &>/dev/null; then
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

  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    pip install fastapi libcst
    pip install --extra-index-url https://test.pypi.org/simple/ llama-models==$TEST_PYPI_VERSION llama-stack==$TEST_PYPI_VERSION $pip_dependencies
  else
    # Re-installing llama-stack in the new conda environment
    if [ -n "$LLAMA_STACK_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: $LLAMA_STACK_DIR${NC}\n" >&2
        exit 1
      fi

      printf "Installing from LLAMA_STACK_DIR: $LLAMA_STACK_DIR\n"
      pip install --no-cache-dir -e "$LLAMA_STACK_DIR"
    else
      pip install --no-cache-dir llama-stack
    fi

    if [ -n "$LLAMA_MODELS_DIR" ]; then
      if [ ! -d "$LLAMA_MODELS_DIR" ]; then
        printf "${RED}Warning: LLAMA_MODELS_DIR is set but directory does not exist: $LLAMA_MODELS_DIR${NC}\n" >&2
        exit 1
      fi

      printf "Installing from LLAMA_MODELS_DIR: $LLAMA_MODELS_DIR\n"
      pip uninstall -y llama-models
      pip install --no-cache-dir -e "$LLAMA_MODELS_DIR"
    fi

    # Install pip dependencies
    if [ -n "$pip_dependencies" ]; then
      printf "Installing pip dependencies: $pip_dependencies\n"
      pip install $pip_dependencies
    fi
  fi
}

ensure_conda_env_python310 "$env_name" "$pip_dependencies"
