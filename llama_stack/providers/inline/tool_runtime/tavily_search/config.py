# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel, Field


class TavilySearchToolConfig(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The Tavily Search API Key",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )
