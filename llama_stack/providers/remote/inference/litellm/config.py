# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Optional

from pydantic import BaseModel, Field


class LitellmConfig(BaseModel):
    openai_api_key: Optional[str] = Field(
        default=None,
        description="The access key to use for openai. Default use environment variable: OPENAI_API_KEY",
    )
    llm_provider: Optional[str] = Field(
        default="openai",
        description="The provider to use. Default use environment variable: LLM_PROVIDER",
    )
