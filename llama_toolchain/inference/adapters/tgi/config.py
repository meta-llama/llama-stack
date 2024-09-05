# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field, field_validator


@json_schema_type
class TGIImplConfig(BaseModel):
    url: str = Field(
        default="https://huggingface.co/inference-endpoints/dedicated",
        description="The URL for the TGI endpoint",
    )
    api_token: Optional[str] = Field(
        default="",
        description="The HF token for Hugging Face Inference Endpoints",
    )
