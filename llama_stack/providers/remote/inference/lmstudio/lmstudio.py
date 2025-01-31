# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_models.sku_list import CoreModelId


# TODO: make sure it follows the same pattern for lmstudio's model id
_MODEL_ALIASES = [
    build_model_alias(
        "meta/llama3-8b-instruct",
        CoreModelId.llama3_8b_instruct.value,
    ),
    build_model_alias(
        "meta/llama3-70b-instruct",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
]
# TODO: Implement LMSTUDIOInferenceAdapter CLASS
