# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class MilvusVectorIOConfig(BaseModel):
    uri: str = Field(description="The URI of the Milvus server")
    token: str | None = Field(description="The token of the Milvus server")
    consistency_level: str = Field(description="The consistency level of the Milvus server", default="Strong")
    kvstore: KVStoreConfig = Field(description="Config for KV store backend")
    embedding_model: str | None = Field(
        default=None,
        description="Optional default embedding model for this provider. If not specified, will use system default.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Optional embedding dimension override. Only needed for models with variable dimensions (e.g., Matryoshka embeddings). If not specified, will auto-lookup from model registry.",
    )

    # This configuration allows additional fields to be passed through to the underlying Milvus client.
    # See the [Milvus](https://milvus.io/docs/install-overview.md) documentation for more details about Milvus in general.
    model_config = ConfigDict(extra="allow")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "uri": "${env.MILVUS_ENDPOINT}",
            "token": "${env.MILVUS_TOKEN}",
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="milvus_remote_registry.db",
            ),
            # Optional: Configure default embedding model for this provider
            # "embedding_model": "all-MiniLM-L6-v2",
            # "embedding_dimension": 384,  # Only needed for variable-dimension models
        }
