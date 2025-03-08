# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_hf_repo_model_entry,
)

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta/Llama-3/Llama-3-8B-Instruct",
        CoreModelId.llama3_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/Llama-3/llama-3-70B-Instruct",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/Llama-3/llama-3_1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/Llama-3/llama-3_2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/Llama-3/llama-3_3-70b-instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
]
