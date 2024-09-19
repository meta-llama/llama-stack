#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

THIS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

set -euo pipefail
set -x

stack_dir=$(dirname $(dirname $THIS_DIR))
models_dir=$(dirname $stack_dir)/llama-models
PYTHONPATH=$models_dir:$stack_dir pytest -p no:warnings --asyncio-mode auto --tb=short
