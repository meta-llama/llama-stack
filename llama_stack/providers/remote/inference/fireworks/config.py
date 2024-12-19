# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class FireworksImplConfig(BaseModel):
    url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="The URL for the Fireworks server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Fireworks.ai API Key",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> Dict[str, Any]:
        return {
            "url": "https://api.fireworks.ai/inference/v1",
            "api_key": "${env.FIREWORKS_API_KEY}",
        }
