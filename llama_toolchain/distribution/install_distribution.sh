#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

error_handler() {
  echo "Error occurred in script at line: ${1}" >&2
  exit 1
}

# Set up the error trap
trap 'error_handler ${LINENO}' ERR

ensure_conda_env_python310() {
  local env_name="$1"
  local pip_dependencies="$2"
  local python_version="3.10"

  # Check if conda command is available
  if ! command -v conda &>/dev/null; then
    echo -e "${RED}Error: conda command not found. Is Conda installed and in your PATH?${NC}" >&2
    exit 1
  fi

  # Check if the environment exists
  if conda env list | grep -q "^${env_name} "; then
    echo "Conda environment '${env_name}' exists. Checking Python version..."

    # Check Python version in the environment
    current_version=$(conda run -n "${env_name}" python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

    if [ "$current_version" = "$python_version" ]; then
      echo "Environment '${env_name}' already has Python ${python_version}. No action needed."
    else
      echo "Updating environment '${env_name}' to Python ${python_version}..."
      conda install -n "${env_name}" python="${python_version}" -y
    fi
  else
    echo "Conda environment '${env_name}' does not exist. Creating with Python ${python_version}..."
    conda create -n "${env_name}" python="${python_version}" -y
  fi

  # Install pip dependencies
  if [ -n "$pip_dependencies" ]; then
    echo "Installing pip dependencies: $pip_dependencies"
    conda run -n "${env_name}" pip install $pip_dependencies
  fi

  # Re-installing llama-toolchain in the new conda environment
  if git rev-parse --is-inside-work-tree &> /dev/null; then
    repo_root=$(git rev-parse --show-toplevel)
    cd "$repo_root"
    conda run -n "${env_name}" pip install -e .
  else
    echo -e "${RED}Not inside a Git repository. Please re-run from within llama-toolchain repository.${NC}"
    exit 1
  fi
}

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <environment_name> <pip_dependencies>" >&2
  echo "Example: $0 my_env 'numpy pandas scipy'" >&2
  exit 1
fi

env_name="$1"
pip_dependencies="$2"

ensure_conda_env_python310 "$env_name" "$pip_dependencies"
