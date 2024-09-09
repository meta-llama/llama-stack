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
    hf_namespace: Optional[str] = Field(
        default=None,
        description="The username/organization name for the Hugging Face Inference Endpoint",
    )
    hf_endpoint_name: Optional[str] = Field(
        default=None,
        description="The name of the Hugging Face Inference Endpoint",
    )

    def is_inference_endpoint(self) -> bool:
        return self.hf_namespace is not None and self.hf_endpoint_name is not None

    def is_local_tgi(self) -> bool:
        return self.url is not None and self.url.startswith("http://localhost")
