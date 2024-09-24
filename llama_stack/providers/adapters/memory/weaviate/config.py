# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field

class WeaviateRequestProviderData(BaseModel):
    # if there _is_ provider data, it must specify the API KEY
    # if you want it to be optional, use Optional[str]
    weaviate_api_key: str

@json_schema_type
class WeaviateConfig(BaseModel):
    url: str = Field(default="http://localhost:8080")
    collection: str = Field(default="MemoryBank")
