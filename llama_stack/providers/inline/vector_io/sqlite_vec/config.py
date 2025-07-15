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


class SQLiteVectorIOConfig(BaseModel):
    db_path: str = Field(description="Path to the SQLite database file")
    kvstore: KVStoreConfig = Field(description="Config for KV store backend (SQLite only for now)")
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + "sqlite_vec.db",
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="sqlite_vec_registry.db",
            ),
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
