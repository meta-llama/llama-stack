# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class TogetherImplConfig(BaseModel):
    url: str = Field(
        default="https://api.together.xyz/v1",
        description="The URL for the Together AI server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Together AI API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.together.xyz/v1",
            "api_key": "${env.TOGETHER_API_KEY}",
        }
