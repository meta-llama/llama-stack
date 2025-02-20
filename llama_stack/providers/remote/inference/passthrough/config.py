# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PassthroughImplConfig(BaseModel):
    url: str = Field(
        default=None,
        description="The URL for the passthrough endpoint",
    )

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API Key for the passthrouth endpoint",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "${env.PASSTHROUGH_URL}",
            "api_key": "${env.PASSTHROUGH_API_KEY}",
        }
