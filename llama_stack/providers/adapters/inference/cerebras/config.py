# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class CerebrasImplConfig(BaseModel):
    base_url: str = Field(
        default=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai"),
        description="Base URL for the Cerebras API",
    )
    api_key: Optional[str] = Field(
        default=os.environ.get("CEREBRAS_API_KEY"),
        description="Cerebras API Key",
    )
