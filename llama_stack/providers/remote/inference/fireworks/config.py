# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class FireworksImplConfig(BaseModel):
    url: str = Field(
        default="https://api.fireworks.ai/inference",
        description="The URL for the Fireworks server",
    )
    api_key: str = Field(
        default="",
        description="The Fireworks.ai API Key",
    )
