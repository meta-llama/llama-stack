# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class MilvusVectorIOConfig(BaseModel):
    db_path: str

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> Dict[str, Any]:
        return {"db_path": "${env.MILVUS_DB_PATH}"}
