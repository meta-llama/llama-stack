# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class PGVectorVectorIOConfig(BaseModel):
    host: str | None = Field(default="localhost")
    port: int | None = Field(default=5432)
    db: str | None = Field(default="postgres")
    user: str | None = Field(default="postgres")
    password: str | None = Field(default="mysecretpassword")
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    @classmethod
    def sample_run_config(
        cls,
        host: str = "${env.PGVECTOR_HOST:=localhost}",
        port: int | str = "${env.PGVECTOR_PORT:=5432}",
        db: str = "${env.PGVECTOR_DB}",
        user: str = "${env.PGVECTOR_USER}",
        password: str = "${env.PGVECTOR_PASSWORD}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "host": host,
            "port": port,
            "db": db,
            "user": user,
            "password": password,
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
