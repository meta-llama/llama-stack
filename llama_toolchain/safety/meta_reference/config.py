# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

from llama_models.sku_list import CoreModelId, safety_models

from pydantic import BaseModel, validator


class LlamaGuardShieldConfig(BaseModel):
    model: str = "Llama-Guard-3-8B"
    excluded_categories: List[str] = []
    disable_input_check: bool = False
    disable_output_check: bool = False

    @validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = [
            m.descriptor()
            for m in safety_models()
            if m.core_model_id == CoreModelId.llama_guard_3_8b
        ]
        if model not in permitted_models:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {permitted_models}"
            )
        return model


class PromptGuardShieldConfig(BaseModel):
    model: str = "Prompt-Guard-86M"

    @validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = [
            m.descriptor()
            for m in safety_models()
            if m.core_model_id == CoreModelId.prompt_guard_86m
        ]
        if model not in permitted_models:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {permitted_models}"
            )
        return model


class SafetyConfig(BaseModel):
    llama_guard_shield: Optional[LlamaGuardShieldConfig] = None
    prompt_guard_shield: Optional[PromptGuardShieldConfig] = None
