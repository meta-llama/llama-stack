# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaImplConfig(BaseModel):
    url: str = DEFAULT_OLLAMA_URL
    refresh_models: bool = Field(default=False, description="refresh and re-register models periodically")
    refresh_models_interval: int = Field(default=300, description="interval in seconds to refresh models")

    @classmethod
    def sample_run_config(cls, url: str = "${env.OLLAMA_URL:=http://localhost:11434}", **kwargs) -> dict[str, Any]:
        return {
            "url": url,
        }
