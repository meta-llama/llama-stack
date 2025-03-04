# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field, SecretStr


@json_schema_type
class CentMLImplConfig(BaseModel):
    url: str = Field(
        default="https://api.centml.org/openai/v1",
        description="The CentML API server URL",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="The CentML API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "url": "https://api.centml.org/openai/v1",
            "api_key": "${env.CENTML_API_KEY}",
        }
