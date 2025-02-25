# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class GroqConfig(BaseModel):
    api_key: Optional[str] = Field(
        # The Groq client library loads the GROQ_API_KEY environment variable by default
        default=None,
        description="The Groq API key",
    )
