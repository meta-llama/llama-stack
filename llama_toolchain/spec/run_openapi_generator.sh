#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

set -x

PYTHONPATH=/data/users/rsm/llama-models:/data/users/rsm/llama-toolchain:/data/users/rsm/llama-agentic-system:../../../oss-ops:../.. python -m toolchain.spec.generate
