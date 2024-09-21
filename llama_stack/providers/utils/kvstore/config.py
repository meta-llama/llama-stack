# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import Annotated


class KVStoreType(Enum):
    redis = "redis"
    sqlite = "sqlite"
    pgvector = "pgvector"


class CommonConfig(BaseModel):
    namespace: Optional[str] = Field(
        default=None,
        description="All keys will be prefixed with this namespace",
    )


class RedisKVStoreImplConfig(CommonConfig):
    type: Literal[KVStoreType.redis.value] = KVStoreType.redis.value
    host: str = "localhost"
    port: int = 6379


class SqliteKVStoreImplConfig(CommonConfig):
    type: Literal[KVStoreType.sqlite.value] = KVStoreType.sqlite.value
    db_path: str = Field(
        description="File path for the sqlite database",
    )


class PGVectorKVStoreImplConfig(CommonConfig):
    type: Literal[KVStoreType.pgvector.value] = KVStoreType.pgvector.value
    host: str = "localhost"
    port: int = 5432
    db: str = "llamastack"
    user: str
    password: Optional[str] = None


KVStoreConfig = Annotated[
    Union[RedisKVStoreImplConfig, SqliteKVStoreImplConfig, PGVectorKVStoreImplConfig],
    Field(discriminator="type"),
]
