# Copyright (c) Nutanix, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class NutanixImplConfig(BaseModel):
    url: str = Field(
        default="https://ai.nutanix.com/api/v1",
        description="The URL of the Nutanix AI Endpoint",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to the Nutanix AI Endpoint",
    )

    @classmethod
    def sample_run_config(cls) -> Dict[str, Any]:
        return {
            "url": "https://ai.nutanix.com/api/v1",
            "api_key": "${env.NUTANIX_API_KEY}",
        }
