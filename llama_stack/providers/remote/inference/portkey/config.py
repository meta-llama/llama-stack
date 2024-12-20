# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.portkey.ai/v1"


@json_schema_type
class PortkeyImplConfig(BaseModel):
    base_url: str = Field(
        default=os.environ.get("PORTKEY_BASE_URL", DEFAULT_BASE_URL),
        description="Base URL for the Portkey API",
    )
    api_key: Optional[str] = Field(
        default=os.environ.get("PORTKEY_API_KEY"),
        description="Portkey API Key",
    )
    virtual_key: Optional[str] = Field(
        default=os.environ.get("PORTKEY_VIRTUAL_KEY"),
        description="Portkey Virtual Key",
    )
    config: Optional[str] = Field(
        default=os.environ.get("PORTKEY_CONFIG_ID"),
        description="Portkey Config ID",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> Dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": "${env.PORTKEY_API_KEY}",
        }
