# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PGVectorVectorIOConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str = Field(default="postgres")
    user: str = Field(default="postgres")
    password: str = Field(default="mysecretpassword")
