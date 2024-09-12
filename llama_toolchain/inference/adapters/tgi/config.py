# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class TGIImplConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the local TGI endpoint (e.g., http://localhost:8080)",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="The HF token for Hugging Face Inference Endpoints (will default to locally saved token if not provided)",
    )
    hf_endpoint_name: Optional[str] = Field(
        default=None,
        description="The name of the Hugging Face Inference Endpoint : can be either in the format of '{namespace}/{endpoint_name}' (namespace can be the username or organization name) or just '{endpoint_name}' if logged into the same account as the namespace",
    )

    def is_inference_endpoint(self) -> bool:
        return self.hf_endpoint_name is not None
