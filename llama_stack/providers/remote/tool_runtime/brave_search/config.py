# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class BraveSearchToolConfig(BaseModel):
    api_key: SecretStr | None = Field(
        default=None,
        description="The Brave Search API Key",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.BRAVE_SEARCH_API_KEY:=}",
            "max_results": 3,
        }
