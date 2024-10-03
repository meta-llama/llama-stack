# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Optional

from llama_models.sku_list import CoreModelId, safety_models

from pydantic import BaseModel, validator


class MetaReferenceShieldType(Enum):
    llama_guard = "llama_guard"
    code_scanner_guard = "code_scanner_guard"
    injection_shield = "injection_shield"
    jailbreak_shield = "jailbreak_shield"


class LlamaGuardShieldConfig(BaseModel):
    model: str = "Llama-Guard-3-1B"
    excluded_categories: List[str] = []
    disable_input_check: bool = False
    disable_output_check: bool = False

    @validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = [
            m.descriptor()
            for m in safety_models()
            if (
                m.core_model_id
                in {
                    CoreModelId.llama_guard_3_8b,
                    CoreModelId.llama_guard_3_1b,
                    CoreModelId.llama_guard_3_11b_vision,
                }
            )
        ]
        if model not in permitted_models:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {permitted_models}"
            )
        return model


class SafetyConfig(BaseModel):
    llama_guard_shield: Optional[LlamaGuardShieldConfig] = None
    enable_prompt_guard: Optional[bool] = False
