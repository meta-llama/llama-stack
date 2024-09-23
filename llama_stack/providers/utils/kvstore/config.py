# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
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


KVStoreConfig = Annotated[
    Union[RedisKVStoreConfig, SqliteKVStoreConfig, PostgresKVStoreConfig],
    Field(discriminator="type", default=KVStoreType.sqlite.value),
]
