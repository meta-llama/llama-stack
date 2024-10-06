# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field, field_validator

from llama_stack.providers.utils.inference import supported_inference_models


@json_schema_type
class VLLMConfig(BaseModel):
    """Configuration for the vLLM inference provider."""

    model: str = Field(
        default="Llama3.1-8B-Instruct",
        description="Model descriptor from `llama model list`",
    )
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of tensor parallel replicas (number of GPUs to use).",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = supported_inference_models()
        if model not in permitted_models:
            model_list = "\n\t".join(permitted_models)
            raise ValueError(
                f"Unknown model: `{model}`. Choose from [\n\t{model_list}\n]"
            )
        return model
