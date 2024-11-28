# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR


class KVStoreType(Enum):
    redis = "redis"
    sqlite = "sqlite"
    postgres = "postgres"
    mongodb = "mongodb"


class CommonConfig(BaseModel):
    namespace: Optional[str] = Field(
        default=None,
        description="All keys will be prefixed with this namespace",
    )


class RedisKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.redis.value] = KVStoreType.redis.value
    host: str = "localhost"
    port: int = 6379

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"

    @classmethod
    def sample_run_config(cls):
        return {
            "type": "redis",
            "namespace": None,
            "host": "${env.REDIS_HOST:localhost}",
            "port": "${env.REDIS_PORT:6379}",
        }


class SqliteKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.sqlite.value] = KVStoreType.sqlite.value
    db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "kvstore.db").as_posix(),
        description="File path for the sqlite database",
    )

    @classmethod
    def sample_run_config(
        cls, __distro_dir__: str = "runtime", db_name: str = "kvstore.db"
    ):
        return {
            "type": "sqlite",
            "namespace": None,
            "db_path": "${env.SQLITE_STORE_DIR:~/.llama/"
            + __distro_dir__
            + "}/"
            + db_name,
        }


class PostgresKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.postgres.value] = KVStoreType.postgres.value
    host: str = "localhost"
    port: int = 5432
    db: str = "llamastack"
    user: str
    password: Optional[str] = None
    table_name: str = "llamastack_kvstore"

    @classmethod
    def sample_run_config(cls, table_name: str = "llamastack_kvstore"):
        return {
            "type": "postgres",
            "namespace": None,
            "host": "${env.POSTGRES_HOST:localhost}",
            "port": "${env.POSTGRES_PORT:5432}",
            "db": "${env.POSTGRES_DB}",
            "user": "${env.POSTGRES_USER}",
            "password": "${env.POSTGRES_PASSWORD}",
            "table_name": "${env.POSTGRES_TABLE_NAME:" + table_name + "}",
        }

    @classmethod
    @field_validator("table_name")
    def validate_table_name(cls, v: str) -> str:
        # PostgreSQL identifiers rules:
        # - Must start with a letter or underscore
        # - Can contain letters, numbers, and underscores
        # - Maximum length is 63 bytes
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid table name. Must start with letter or underscore and contain only letters, numbers, and underscores"
            )
        if len(v) > 63:
            raise ValueError("Table name must be less than 63 characters")
        return v


class MongoDBKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.mongodb.value] = KVStoreType.mongodb.value
    host: str = "localhost"
    port: int = 27017
    db: str = "llamastack"
    user: str = None
    password: Optional[str] = None
    collection_name: str = "llamastack_kvstore"

    @classmethod
    def sample_run_config(cls, collection_name: str = "llamastack_kvstore"):
        return {
            "type": "mongodb",
            "namespace": None,
            "host": "${env.MONGODB_HOST:localhost}",
            "port": "${env.MONGODB_PORT:5432}",
            "db": "${env.MONGODB_DB}",
            "user": "${env.MONGODB_USER}",
            "password": "${env.MONGODB_PASSWORD}",
            "collection_name": "${env.MONGODB_COLLECTION_NAME:" + collection_name + "}",
        }


KVStoreConfig = Annotated[
    Union[RedisKVStoreConfig, SqliteKVStoreConfig, PostgresKVStoreConfig, MongoDBKVStoreConfig],
    Field(discriminator="type", default=KVStoreType.sqlite.value),
]
