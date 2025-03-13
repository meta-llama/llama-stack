# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel


class ChromaVectorIOConfig(BaseModel):
    db_path: str

    @classmethod
    def sample_run_config(cls, db_path: str = "${env.CHROMADB_PATH}", **kwargs: Any) -> Dict[str, Any]:
        return {"db_path": db_path}
