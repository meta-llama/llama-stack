# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_hf_repo_model_entry,
)

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "Llama-3.3-70B-Instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "Llama-4-Scout-17B-16E-Instruct-FP8",
        CoreModelId.llama4_scout_17b_16e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        CoreModelId.llama4_maverick_17b_128e_instruct.value,
    ),
]
