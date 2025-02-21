# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from pydantic import BaseModel, Field

from llama_models.schema_utils import json_schema_type


@json_schema_type
class FiddlecubeSafetyConfig(BaseModel):
    api_url: str = "https://api.fiddlecube.ai/api"
    excluded_categories: List[str] = Field(default_factory=list)
