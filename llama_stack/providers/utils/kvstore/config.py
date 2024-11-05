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


class SqliteKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.sqlite.value] = KVStoreType.sqlite.value
    db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "kvstore.db").as_posix(),
        description="File path for the sqlite database",
    )


class PostgresKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.postgres.value] = KVStoreType.postgres.value
    host: str = "localhost"
    port: int = 5432
    db: str = "llamastack"
    user: str
    password: Optional[str] = None
    table_name: str = "llamastack_kvstore"

    @field_validator("table_name")
    def validate_table_name(self, v: str) -> str:
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


KVStoreConfig = Annotated[
    Union[RedisKVStoreConfig, SqliteKVStoreConfig, PostgresKVStoreConfig],
    Field(discriminator="type", default=KVStoreType.sqlite.value),
]
