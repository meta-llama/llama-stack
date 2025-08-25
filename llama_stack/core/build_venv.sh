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
UV_SYSTEM_PYTHON=${UV_SYSTEM_PYTHON:-}
VIRTUAL_ENV=${VIRTUAL_ENV:-}

set -euo pipefail

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

# Usage function
usage() {
  echo "Usage: $0 --env-name <env_name> --normal-deps <pip_dependencies> [--external-provider-deps <external_provider_deps>] [--optional-deps <special_pip_deps>]"
  echo "Example: $0 --env-name mybuild --normal-deps '{\"default\": [\"numpy\", \"pandas\"]}' --external-provider-deps '{\"vector_db\": [\"chromadb\"]}' --optional-deps '{\"special\": [\"bar\"]}'"
  exit 1
}

# Parse arguments
env_name=""
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
    --normal-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --normal-deps requires a JSON object" >&2
        usage
      fi
      normal_deps="$2"
      shift 2
      ;;
    --external-provider-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --external-provider-deps requires a JSON object" >&2
        usage
      fi
      external_provider_deps="$2"
      shift 2
      ;;
    --optional-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --optional-deps requires a JSON object" >&2
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
if [[ -z "$env_name" || -z "$normal_deps" ]]; then
  echo "Error: --env-name and --normal-deps are required." >&2
  usage
fi

if [ -n "$LLAMA_STACK_DIR" ]; then
  echo "Using llama-stack-dir=$LLAMA_STACK_DIR"
fi
if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  echo "Using llama-stack-client-dir=$LLAMA_STACK_CLIENT_DIR"
fi

ENVNAME=""

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

# Function to install dependencies from JSON object
install_deps_from_json() {
  local json_deps="$1"
  local dep_type="$2"

  if [ -n "$json_deps" ]; then
    if [ "$dep_type" = "optional" ]; then
      # For optional deps, process each spec separately to preserve flags like --no-deps
      local last_provider=""
      echo "$json_deps" | jq -r 'to_entries[] | .key as $k | .value[] | "\($k)\t\(.)"' | while IFS=$'\t' read -r provider spec; do
        if [ -n "$spec" ]; then
          if [ "$provider" != "$last_provider" ]; then
            echo "Installing $dep_type dependencies for provider '$provider'"
            last_provider="$provider"
          fi
          uv pip install $spec
        fi
      done
    else
      # For normal deps, install all at once (no special flags)
      echo "$json_deps" | jq -r 'to_entries[] | "\(.key)\t\(.value | join(" "))"' | while IFS=$'\t' read -r provider deps; do
        if [ -n "$deps" ]; then
          echo "Installing $dep_type dependencies for provider '$provider'"
          uv pip install $deps
        fi
      done
    fi
  fi
}

run() {
  # Use only global variables set by flag parser
  if [ -n "$UV_SYSTEM_PYTHON" ] || [ "$env_name" == "__system__" ]; then
    echo "Installing dependencies in system Python environment"
    export UV_SYSTEM_PYTHON=1
  elif [ "$VIRTUAL_ENV" == "$env_name" ]; then
    echo "Virtual environment $env_name is already active"
  else
    echo "Using virtual environment $env_name"
    uv venv "$env_name"
    source "$env_name/bin/activate"
  fi

  if [ -n "$TEST_PYPI_VERSION" ]; then
    uv pip install fastapi libcst
    uv pip install --extra-index-url https://test.pypi.org/simple/ \
      --index-strategy unsafe-best-match \
      llama-stack=="$TEST_PYPI_VERSION" \
      $normal_deps
    if [ -n "$optional_deps" ]; then
      install_deps_from_json "$optional_deps" "optional"
    fi
    if [ -n "$external_provider_deps" ]; then
      # Install external provider modules (no special flags supported)
      echo "Installing external provider modules"
      echo "$external_provider_deps" | jq -r 'to_entries[] | .value[]' | while read -r part; do
        if [ -n "$part" ]; then
          echo "Installing external provider module: $part"
          uv pip install "$part"
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
        fi
      done
    fi
  else
    if [ -n "$LLAMA_STACK_DIR" ]; then
      # only warn if DIR does not start with "git+"
      if [ ! -d "$LLAMA_STACK_DIR" ] && [[ "$LLAMA_STACK_DIR" != git+* ]]; then
        printf "${RED}Warning: LLAMA_STACK_DIR is set but directory does not exist: %s${NC}\n" "$LLAMA_STACK_DIR" >&2
        exit 1
      fi
      printf "Installing from LLAMA_STACK_DIR: %s\n"  "$LLAMA_STACK_DIR"
      # editable only if LLAMA_STACK_DIR does not start with "git+"
      if [[ "$LLAMA_STACK_DIR" != git+* ]]; then
        EDITABLE="-e"
      else
        EDITABLE=""
      fi
      uv pip install --no-cache-dir --quiet $EDITABLE "$LLAMA_STACK_DIR"
    else
      uv pip install --no-cache-dir --quiet llama-stack
    fi

    if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
      # only warn if DIR does not start with "git+"
      if [ ! -d "$LLAMA_STACK_CLIENT_DIR" ] && [[ "$LLAMA_STACK_CLIENT_DIR" != git+* ]]; then
        printf "${RED}Warning: LLAMA_STACK_CLIENT_DIR is set but directory does not exist: %s${NC}\n" "$LLAMA_STACK_CLIENT_DIR" >&2
        exit 1
      fi
      printf "Installing from LLAMA_STACK_CLIENT_DIR: %s\n" "$LLAMA_STACK_CLIENT_DIR"
      # editable only if LLAMA_STACK_CLIENT_DIR does not start with "git+"
      if [[ "$LLAMA_STACK_CLIENT_DIR" != git+* ]]; then
        EDITABLE="-e"
      else
        EDITABLE=""
      fi
      uv pip install --no-cache-dir --quiet $EDITABLE "$LLAMA_STACK_CLIENT_DIR"
    fi

    printf "Installing pip dependencies\n"
    install_deps_from_json "$normal_deps" "normal"
    if [ -n "$optional_deps" ]; then
      install_deps_from_json "$optional_deps" "optional"
    fi
    if [ -n "$external_provider_deps" ]; then
      install_deps_from_json "$external_provider_deps" "external provider"

      # For external provider deps, also get and install their dependencies
      echo "Getting provider specs and installing dependencies for external providers"
      echo "$external_provider_deps" | jq -r 'to_entries[] | "\(.key) \(.value | join(" "))"' | while read -r provider deps; do
        if [ -n "$deps" ]; then
          echo "Getting provider specs for provider '$provider' dependencies: $deps"
          for dep in $deps; do
            package_name=$(echo "$dep" | sed 's/[<>=!].*//')
            echo "Getting provider spec for module: $package_name"
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
      done
    fi
  fi
}

pre_run_checks "$env_name"
run
