# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class AnthropicProviderDataValidator(BaseModel):
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="API key for Anthropic models",
    )


@json_schema_type
class AnthropicConfig(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="API key for Anthropic models",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.ANTHROPIC_API_KEY}", **kwargs) -> Dict[str, Any]:
        return {
            "api_key": api_key,
        }
