# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel, Field

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaImplConfig(BaseModel):
    url: str = Field(DEFAULT_OLLAMA_URL, json_schema_extra={"user_field": True})

    @classmethod
    def sample_run_config(cls, url: str = "${env.OLLAMA_URL:http://localhost:11434}", **kwargs) -> Dict[str, Any]:
        return {"url": url}
