# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel

DEFAULT_PODMAN_AI_LAB_URL = "http://localhost:10434"


class PodmanAILabImplConfig(BaseModel):
    url: str = DEFAULT_PODMAN_AI_LAB_URL

    @classmethod
    def sample_run_config(
        cls, url: str = "${env.PODMAN_AI_LAB_URL:http://localhost:10434}", **kwargsi
    ) -> Dict[str, Any]:
        return {"url": url}
