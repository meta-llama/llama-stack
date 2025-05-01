# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class GroqProviderDataValidator(BaseModel):
    groq_api_key: str | None = Field(
        default=None,
        description="API key for Groq models",
    )


@json_schema_type
class GroqCompatConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="The Groq API key",
    )

    openai_compat_api_base: str = Field(
        default="https://api.groq.com/openai/v1",
        description="The URL for the Groq API server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.GROQ_API_KEY}", **kwargs) -> dict[str, Any]:
        return {
            "openai_compat_api_base": "https://api.groq.com/openai/v1",
            "api_key": api_key,
        }
