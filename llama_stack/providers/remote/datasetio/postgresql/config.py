# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from pydantic import BaseModel, Field
from typing import Any, Dict
import os

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)


class PostgreSQLDatasetIOConfig(BaseModel):
    kvstore: KVStoreConfig

    pg_host: str = Field(default="localhost")  # os.getenv("POSTGRES_HOST", "127.0.0.1")
    pg_port: int = Field(default=5432)
    # TODO - revise secutiry implications of using env vars for user and password
    pg_user: str = Field(default="postgres")
    pg_password: str = Field(default="fail")
    pg_con_pool_size: int = Field(default=3)
    pg_database: str = Field(default="postgres")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="postgresql_datasetio.db",
            ),
            "pg_host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
            "pg_port": os.getenv("POSTGRES_PORT", 5432),
            "pg_user": os.getenv("POSTGRES_USER", ""),
            "pg_password": os.getenv("POSTGRES_PASSWORD", ""),
            "pg_con_pool_size": os.getenv("POSTGRES_CONN_POOL_SIZE", 3),
            "pg_database": os.getenv("POSTGRES_DATABASE", "postgres"),
        }
