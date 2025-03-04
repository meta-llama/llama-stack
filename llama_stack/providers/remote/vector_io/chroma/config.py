# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel


class ChromaVectorIOConfig(BaseModel):
    url: str

    @classmethod
    def sample_run_config(cls, url: str = "${env.CHROMADB_URL}", **kwargs: Any) -> Dict[str, Any]:
        return {"url": url}
