# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, model_validator


@json_schema_type
class ChromaConfig(BaseModel):
    # You can either specify the url of the chroma server or the path to the local db
    url: Optional[str] = None
    db_path: Optional[str] = None

    @model_validator(mode="after")
    def check_url_or_db_path(self):
        if not (self.url or self.db_path):
            raise ValueError("Either url or db_path must be specified")

    @classmethod
    def sample_config(cls) -> Dict[str, Any]:
        return {"url": "{env.CHROMADB_URL}"}
