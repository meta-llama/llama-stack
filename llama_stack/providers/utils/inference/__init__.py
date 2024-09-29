# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_models.datatypes import *  # noqa: F403
from llama_models.sku_list import all_registered_models


def is_supported_safety_model(model: Model) -> bool:
    if model.quantization_format != CheckpointQuantizationFormat.bf16:
        return False

    model_id = model.core_model_id
    return model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.llama_guard_3_1b,
        CoreModelId.llama_guard_3_11b_vision,
    ]


def supported_inference_models() -> List[str]:
    return [
        m.descriptor()
        for m in all_registered_models()
        if (
            m.model_family in {ModelFamily.llama3_1, ModelFamily.llama3_2}
            or is_supported_safety_model(m)
        )
    ]
