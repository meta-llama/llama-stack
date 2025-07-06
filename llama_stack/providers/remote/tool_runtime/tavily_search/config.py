# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field


class TavilySearchToolConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="The Tavily Search API Key",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )
    timeout: float = Field(
        default=30.0,
        description="HTTP request timeout for the API",
    )
    connect_timeout: float = Field(
        default=10.0,
        description="HTTP connection timeout in seconds for the API",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.TAVILY_SEARCH_API_KEY:=}",
            "max_results": 3,
            "timeout:" 30.0,
            "connect_timeout": 10.0,
        }
