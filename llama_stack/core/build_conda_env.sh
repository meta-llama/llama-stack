#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
PYPI_VERSION=${PYPI_VERSION:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}

set -euo pipefail

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# Usage function
usage() {
  echo "Usage: $0 --env-name <conda_env_name> --build-file-path <build_file_path> --normal-deps <pip_dependencies> [--external-provider-deps <external_provider_deps>] [--optional-deps <special_pip_deps>]"
  echo "Example: $0 --env-name my-conda-env --build-file-path ./my-stack-build.yaml --normal-deps 'numpy pandas scipy' --external-provider-deps 'foo' --optional-deps 'bar'"
  exit 1
}

# Parse arguments
env_name=""
build_file_path=""
normal_deps=""
external_provider_deps=""
optional_deps=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --env-name)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --env-name requires a string value" >&2
        usage
      fi
      env_name="$2"
      shift 2
      ;;
    --build-file-path)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --build-file-path requires a string value" >&2
        usage
      fi
      build_file_path="$2"
      shift 2
      ;;
    --normal-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --normal-deps requires a string value" >&2
        usage
      fi
      normal_deps="$2"
      shift 2
      ;;
    --external-provider-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --external-provider-deps requires a string value" >&2
        usage
      fi
      external_provider_deps="$2"
      shift 2
      ;;
    --optional-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --optional-deps requires a string value" >&2
        usage
      fi
      optional_deps="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

# Check required arguments
if [[ -z "$env_name" || -z "$build_file_path" || -z "$normal_deps" ]]; then
  echo "Error: --env-name, --build-file-path, and --normal-deps are required." >&2
  usage
fi

if [ -n "$LLAMA_STACK_DIR" ]; then
  echo "Using llama-stack-dir=$LLAMA_STACK_DIR"
fi
if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  echo "Using llama-stack-client-dir=$LLAMA_STACK_CLIENT_DIR"
fi

ensure_conda_env_python310() {
  # Use only global variables set by flag parser
  local python_version="3.12"

  if ! is_command_available conda; then
    printf "${RED}Error: conda command not found. Is Conda installed and in your PATH?${NC}" >&2
    exit 1
  fi

  if conda env list | grep -q "^${env_name} "; then
    printf "Conda environment '${env_name}' exists. Checking Python version...\n"
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
  fi

  eval "$(conda shell.bash hook)"
  conda deactivate && conda activate "${env_name}"
  "$CONDA_PREFIX"/bin/pip install uv

  if [ -n "$TEST_PYPI_VERSION" ]; then
    uv pip install fastapi libcst
    uv pip install --extra-index-url https://test.pypi.org/simple/ \
      llama-stack=="$TEST_PYPI_VERSION" \
      "$normal_deps"
    if [ -n "$optional_deps" ]; then
      IFS='#' read -ra parts <<<"$optional_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        uv pip install $part
      done
    fi
    if [ -n "$external_provider_deps" ]; then
      IFS='#' read -ra parts <<<"$external_provider_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        uv pip install "$part"
      done
    fi
  else
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
      uv pip install --no-cache-dir "$SPEC_VERSION"
    fi
    if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
      if [ ! -d "$LLAMA_STACK_CLIENT_DIR" ]; then
        printf "${RED}Warning: LLAMA_STACK_CLIENT_DIR is set but directory does not exist: $LLAMA_STACK_CLIENT_DIR${NC}\n" >&2
        exit 1
      fi
      printf "Installing from LLAMA_STACK_CLIENT_DIR: $LLAMA_STACK_CLIENT_DIR\n"
      uv pip install --no-cache-dir -e "$LLAMA_STACK_CLIENT_DIR"
    fi
    printf "Installing pip dependencies\n"
    uv pip install $normal_deps
    if [ -n "$optional_deps" ]; then
      IFS='#' read -ra parts <<<"$optional_deps"
      for part in "${parts[@]}"; do
        echo "$part"
        uv pip install $part
      done
    fi
    if [ -n "$external_provider_deps" ]; then
      IFS='#' read -ra parts <<<"$external_provider_deps"
      for part in "${parts[@]}"; do
        echo "Getting provider spec for module: $part and installing dependencies"
        package_name=$(echo "$part" | sed 's/[<>=!].*//')
        python3 -c "
import importlib
import sys
try:
    module = importlib.import_module(f'$package_name.provider')
    spec = module.get_provider_spec()
    if hasattr(spec, 'pip_packages') and spec.pip_packages:
        print('\\n'.join(spec.pip_packages))
except Exception as e:
    print(f'Error getting provider spec for $package_name: {e}', file=sys.stderr)
" | uv pip install -r -
      done
    fi
  fi
  mv "$build_file_path" "$CONDA_PREFIX"/llamastack-build.yaml
  echo "Build spec configuration saved at $CONDA_PREFIX/llamastack-build.yaml"
}

ensure_conda_env_python310 "$env_name" "$build_file_path" "$normal_deps" "$optional_deps" "$external_provider_deps"
