#!/bin/sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

echo "Preloading (${INFERENCE_MODEL}, ${SAFETY_MODEL})..."
for model in ${INFERENCE_MODEL} ${SAFETY_MODEL}; do
  echo "Preloading $model..."
  if ! ollama run "$model"; then
    echo "Failed to pull and run $model"
    exit 1
  fi
done

echo "All models pulled successfully"
