# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class DatabricksImplConfig(BaseModel):
    url: str = Field(
        default=None,
        description="The URL for the Databricks model serving endpoint",
    )
    api_token: str = Field(
        default=None,
        description="The Databricks API token",
    )
