# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class GroqImplConfig(BaseModel):
    url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="The URL for the Groq API server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Groq API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.groq.com/openai/v1",
            "api_key": "${env.GROQ_API_KEY}",
        }
