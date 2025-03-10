# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Common variables
LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}
TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}
CONTAINER_BINARY=${CONTAINER_BINARY:-docker}
CONTAINER_OPTS=${CONTAINER_OPTS:-}
PYPI_VERSION=${PYPI_VERSION:-}

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

cleanup() {
  envname="$1"

  set +x
  echo "Cleaning up..."
  conda deactivate
  conda env remove --name "$envname" -y
}

handle_int() {
  if [ -n "$ENVNAME" ]; then
    cleanup "$ENVNAME"
  fi
  exit 1
}

handle_exit() {
  if [ $? -ne 0 ]; then
    echo -e "\033[1;31mABORTING.\033[0m"
    if [ -n "$ENVNAME" ]; then
      cleanup "$ENVNAME"
    fi
  fi
}

setup_cleanup_handlers() {
  trap handle_int INT
  trap handle_exit EXIT

  if is_command_available conda; then
    __conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
    eval "$__conda_setup"
    conda deactivate
  else
    echo "conda is not available"
    exit 1
  fi
}

# check if a command is present
is_command_available() {
  command -v "$1" &>/dev/null
}
