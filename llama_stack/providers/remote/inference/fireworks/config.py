# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class FireworksImplConfig(BaseModel):
    url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="The URL for the Fireworks server",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="The Fireworks.ai API Key",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.FIREWORKS_API_KEY}", **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.fireworks.ai/inference/v1",
            "api_key": api_key,
        }
