#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail
set -x

cd $(git rev-parse --show-toplevel)

MASTER_HOST=$1
RUN_ID=$2
CKPT_DIR=$3
QUANT_CKPT_DIR=$4
TOKENIZER_PATH=$5
NNODES=$6
NPROC=$7

echo $MASTER_HOST, $RUN_ID, $CKPT_DIR, $QUANT_CKPT_DIR

NCCL_NET=Socket NCCL_SOCKET_IFNAME=eth TIKTOKEN_CACHE_DIR="" \
  torchrun \
   --nnodes=$NNODES --nproc_per_node=$NPROC \
   --rdzv_id=$RUN_ID \
   --rdzv_conf='timeout=120' \
   --rdzv_backend=c10d \
   --rdzv_endpoint="${MASTER_HOST}:29502" \
   quantize_checkpoint.py $CKPT_DIR $TOKENIZER_PATH $QUANT_CKPT_DIR
