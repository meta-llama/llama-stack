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
    url: str = Field(
        description="The URL for the TGI endpoint (e.g. 'http://localhost:8080')",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="A bearer token if your TGI endpoint is protected.",
    )


@json_schema_type
class InferenceEndpointImplConfig(BaseModel):
    endpoint_name: str = Field(
        description="The name of the Hugging Face Inference Endpoint in the format of '{namespace}/{endpoint_name}' (e.g. 'my-cool-org/meta-llama-3-1-8b-instruct-rce'). Namespace is optional and will default to the user account if not provided.",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="Your Hugging Face user access token (will default to locally saved token if not provided)",
    )


@json_schema_type
class InferenceAPIImplConfig(BaseModel):
    huggingface_repo: str = Field(
        description="The model ID of the model on the Hugging Face Hub (e.g. 'meta-llama/Meta-Llama-3.1-70B-Instruct')",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="Your Hugging Face user access token (will default to locally saved token if not provided)",
    )
