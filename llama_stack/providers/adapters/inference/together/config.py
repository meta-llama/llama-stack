# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_models.schema_utils import json_schema_type

from llama_stack.distribution.request_headers import annotate_header


class TogetherHeaderExtractor(BaseModel):
    api_key: annotate_header(
        "X-LlamaStack-Together-ApiKey", str, "The API Key for the request"
    )


@json_schema_type
class TogetherImplConfig(BaseModel):
    url: str = Field(
        default="https://api.together.xyz/v1",
        description="The URL for the Together AI server",
    )
    api_key: str = Field(
        default="",
        description="The Together AI API Key",
    )
