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
        "sambanova/Meta-Llama-3.1-8B-Instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Meta-Llama-3.1-405B-Instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Meta-Llama-3.2-1B-Instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Meta-Llama-3.2-3B-Instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Meta-Llama-3.3-70B-Instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Llama-3.2-11B-Vision-Instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Llama-3.2-90B-Vision-Instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Llama-4-Scout-17B-16E-Instruct",
        CoreModelId.llama4_scout_17b_16e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Llama-4-Maverick-17B-128E-Instruct",
        CoreModelId.llama4_maverick_17b_128e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "sambanova/Meta-Llama-Guard-3-8B",
        CoreModelId.llama_guard_3_8b.value,
    ),
]
