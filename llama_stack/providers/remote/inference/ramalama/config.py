# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel

DEFAULT_RAMALAMA_URL = "http://localhost:8080"


class RamalamaImplConfig(BaseModel):
    url: str = DEFAULT_RAMALAMA_URL

    @classmethod
    def sample_run_config(cls, url: str = "${env.RAMALAMA_URL:http://localhost:8080}", **kwargs) -> Dict[str, Any]:
        return {"url": url}
