# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class FireworksProviderDataValidator(BaseModel):
    fireworks_api_key: str | None = Field(
        default=None,
        description="API key for Fireworks models",
    )


@json_schema_type
class FireworksCompatConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="The Fireworks API key",
    )

    openai_compat_api_base: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="The URL for the Fireworks API server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.FIREWORKS_API_KEY}", **kwargs) -> dict[str, Any]:
        return {
            "openai_compat_api_base": "https://api.fireworks.ai/inference/v1",
            "api_key": api_key,
        }
