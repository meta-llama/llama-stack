# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel


class ChromaInlineImplConfig(BaseModel):
    db_path: str

    @classmethod
    def sample_config(cls) -> Dict[str, Any]:
        return {"db_path": "{env.CHROMADB_PATH}"}
