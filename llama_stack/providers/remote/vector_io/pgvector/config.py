# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PGVectorVectorIOConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str = Field(default="postgres")
    user: str = Field(default="postgres")
    password: str = Field(default="mysecretpassword")

    @classmethod
    def sample_run_config(
        cls,
        host: str = "${env.PGVECTOR_HOST:localhost}",
        port: int = "${env.PGVECTOR_PORT:5432}",
        db: str = "${env.PGVECTOR_DB}",
        user: str = "${env.PGVECTOR_USER}",
        password: str = "${env.PGVECTOR_PASSWORD}",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"host": host, "port": port, "db": db, "user": user, "password": password}
