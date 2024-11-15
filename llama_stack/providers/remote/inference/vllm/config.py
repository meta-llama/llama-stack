# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class VLLMInferenceAdapterConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the vLLM model serving endpoint",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    api_token: Optional[str] = Field(
        default="fake",
        description="The API token",
    )

    @classmethod
    def sample_dict(cls):
        # TODO: we may need two modes, one for conda and one for docker
        return {
            "url": "${env.VLLM_URL:http://host.docker.internal:5100/v1}",
            "max_tokens": "${env.VLLM_MAX_TOKENS:4096}",
            "api_token": "${env.VLLM_API_TOKEN:fake}",
        }
