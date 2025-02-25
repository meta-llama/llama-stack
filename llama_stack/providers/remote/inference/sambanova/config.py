# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class SambaNovaImplConfig(BaseModel):
    url: str = Field(
        default="https://api.sambanova.ai/v1",
        description="The URL for the SambaNova AI server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The SambaNova.ai API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.sambanova.ai/v1",
            "api_key": "${env.SAMBANOVA_API_KEY}",
        }
