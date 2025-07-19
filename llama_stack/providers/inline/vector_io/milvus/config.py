# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class MilvusVectorIOConfig(BaseModel):
    db_path: str
    kvstore: KVStoreConfig = Field(description="Config for KV store backend (SQLite only for now)")
    consistency_level: str = Field(description="The consistency level of the Milvus server", default="Strong")
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "db_path": "${env.MILVUS_DB_PATH:=" + __distro_dir__ + "}/" + "milvus.db",
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="milvus_registry.db",
            ),
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
