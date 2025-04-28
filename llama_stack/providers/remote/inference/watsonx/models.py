# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.utils.inference.model_registry import build_hf_repo_model_entry

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta-llama/llama-3-3-70b-instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-2-13b-chat",
        CoreModelId.llama2_13b.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-3-2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-guard-3-11b-vision",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]
