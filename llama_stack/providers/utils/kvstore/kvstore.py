# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .api import *  # noqa: F403
from .config import *  # noqa: F403


def kvstore_dependencies():
    return ["aiosqlite", "psycopg2-binary", "redis"]


async def kvstore_impl(config: KVStoreConfig) -> KVStore:
    if config.type == KVStoreType.redis:
        from .redis import RedisKVStoreImpl

        impl = RedisKVStoreImpl(config)
    elif config.type == KVStoreType.sqlite:
        from .sqlite import SqliteKVStoreImpl

        impl = SqliteKVStoreImpl(config)
    elif config.type == KVStoreType.pgvector:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown kvstore type {config.type}")

    await impl.initialize()
    return impl
