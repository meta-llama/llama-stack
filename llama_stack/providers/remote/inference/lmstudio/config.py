# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel


DEFAULT_LMSTUDIO_URL = "http://localhost:12345"


class LMSTUDIOImplConfig(BaseModel):
    url: str = DEFAULT_LMSTUDIO_URL

    @classmethod
    def sample_run_config(
        cls, url: str = "${env.LMSTUDIO_URL:http://localhost:12345}", **kwargs
    ) -> Dict[str, Any]:
        return {"url": url}
