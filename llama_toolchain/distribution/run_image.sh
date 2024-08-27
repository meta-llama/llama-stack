#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

podman run -it -p 8001:8001 -v ~/.llama/test.yaml:/app/test.yaml test-image --yaml_config /app/test.yaml --disable-ipv6
