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
from llama_stack.providers.utils.vector_io.embedding_config import EmbeddingConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class MilvusVectorIOConfig(BaseModel):
    db_path: str
    kvstore: KVStoreConfig = Field(description="Config for KV store backend (SQLite only for now)")
    consistency_level: str = Field(description="The consistency level of the Milvus server", default="Strong")
    embedding: EmbeddingConfig | None = Field(
        default=None,
        description="Default embedding configuration for this provider. When specified, vector databases created with this provider will use these embedding settings as defaults.",
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
            # "embedding": {
            #     "model": "${env.MILVUS_EMBEDDING_MODEL:=all-MiniLM-L6-v2}",
            #     "dimensions": 384
            # },
        }
