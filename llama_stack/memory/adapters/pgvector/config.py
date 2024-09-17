# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class PGVectorConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str
    user: str
    password: str
