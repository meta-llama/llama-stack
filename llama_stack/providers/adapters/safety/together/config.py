# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


class TogetherProviderDataValidator(BaseModel):
    together_api_key: str


@json_schema_type
class TogetherSafetyConfig(BaseModel):
    url: str = Field(
        default="https://api.together.xyz/v1",
        description="The URL for the Together AI server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Together AI API Key (default for the distribution, if any)",
    )
