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
class WeaviateVectorIOConfig(BaseModel):
    host: str | None = Field(default="localhost")
    port: int | None = Field(default=8080)
    weaviate_api_key: str | None = Field(description="The API key for the Weaviate instance", default=None)
    weaviate_cluster_url: str | None = Field(description="The URL of the Weaviate cluster", default=None)
    kvstore: KVStoreConfig | None = Field(description="Config for KV store backend (SQLite only for now)", default=None)

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        host: str = "${env.WEAVIATE_HOST:=localhost}",
        port: int = "${env.WEAVIATE_PORT:=8080}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "host": "${env.WEAVIATE_HOST:=localhost}",
            "port": "${env.WEAVIATE_PORT:=8080}",
            "weaviate_api_key": None,
            "weaviate_cluster_url": None,
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="weaviate_registry.db",
            ),
        }
