#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

TMPDIR=$(mktemp -d)
echo "Using temporary directory: $TMPDIR"

rootdir=$(git rev-parse --show-toplevel)

files_to_copy=("toolchain/spec/openapi*" "llama_models.llama3_1.api.datatypes.py" "toolchain/inference/api/*.py" "agentic_system/api/*.py" "toolchain/common/*.py" "toolchain/dataset/api/*.py" "toolchain/evaluations/api/*.py" "toolchain/reward_scoring/api/*.py" "toolchain/post_training/api/*.py" "toolchain/safety/api/*.py")
for file in "${files_to_copy[@]}"; do
    relpath="$file"
    set -x
    mkdir -p "$TMPDIR/$(dirname $relpath)"
    eval cp "$rootdir/$relpath" "$TMPDIR/$(dirname $relpath)"
    set +x
done

cd "$TMPDIR"
zip -r output.zip .

echo "Zip at: $TMPDIR/output.zip"
