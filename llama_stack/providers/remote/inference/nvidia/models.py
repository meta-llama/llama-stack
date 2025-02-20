# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_hf_repo_model_alias,
)

_MODEL_ALIASES = [
    build_hf_repo_model_alias(
        "meta/llama3-8b-instruct",
        CoreModelId.llama3_8b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama3-70b-instruct",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_alias(
        "meta/llama-3.2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    # TODO(mf): how do we handle Nemotron models?
    # "Llama3.1-Nemotron-51B-Instruct" -> "meta/llama-3.1-nemotron-51b-instruct",
]
