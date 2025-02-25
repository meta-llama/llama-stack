#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

VERSION="$1"

set -euo pipefail
set -x

uv pip install -U --extra-index-url https://test.pypi.org/simple \
  llama-stack==$VERSION llama-models==$VERSION llama-stack-client==$VERSION
