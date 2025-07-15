# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class LlamaCppProviderDataValidator(BaseModel):
    llamacpp_api_key: str | None = Field(
        default=None,
        description="API key for llama.cpp server (optional for local servers)",
    )


@json_schema_type
class LlamaCppImplConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="The llama.cpp server API key (optional for local servers)",
    )

    openai_compat_api_base: str = Field(
        default="http://localhost:8080",
        description="The URL for the llama.cpp server with OpenAI-compatible API",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.LLAMACPP_API_KEY:=}") -> dict[str, Any]:
        return {
            "openai_compat_api_base": "${env.LLAMACPP_URL:=http://localhost:8080}",
            "api_key": api_key,
        }
