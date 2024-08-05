# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field
from strong_typing.schema import json_schema_type


@json_schema_type
class OllamaImplConfig(BaseModel):
    model: str = Field(..., description="The name of the model in ollama catalog")
    url: str = Field(..., description="The URL for the ollama server")
