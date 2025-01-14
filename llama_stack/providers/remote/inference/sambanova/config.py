# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class SambanovaImplConfig(BaseModel):
    url: str = Field(
        default="https://api.sambanova.ai/v1",
        description="The URL for the SambaNova API server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The SambaNova API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.sambanova.ai/v1",
            "api_key": "${env.SAMBANOVA_API_KEY}",
        }
