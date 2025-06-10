# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from .api import KVStore
from .config import KVStoreConfig, KVStoreType


def kvstore_dependencies():
    """
    Returns all possible kvstore dependencies for registry/provider specifications.

    NOTE: For specific kvstore implementations, use config.pip_packages instead.
    This function returns the union of all dependencies for cases where the specific
    kvstore type is not known at declaration time (e.g., provider registries).
    """
    return ["aiosqlite", "psycopg2-binary", "redis", "pymongo"]


class InmemoryKVStoreImpl(KVStore):
    def __init__(self):
        self._store = {}

    async def initialize(self) -> None:
        pass

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [self._store[key] for key in self._store.keys() if key >= start_key and key < end_key]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        return [key for key in self._store.keys() if key >= start_key and key < end_key]

    async def delete(self, key: str) -> None:
        del self._store[key]


async def kvstore_impl(config: KVStoreConfig) -> KVStore:
    if config.type == KVStoreType.redis.value:
        from .redis import RedisKVStoreImpl

        impl = RedisKVStoreImpl(config)
    elif config.type == KVStoreType.sqlite.value:
        from .sqlite import SqliteKVStoreImpl

        impl = SqliteKVStoreImpl(config)
    elif config.type == KVStoreType.postgres.value:
        from .postgres import PostgresKVStoreImpl

        impl = PostgresKVStoreImpl(config)
    elif config.type == KVStoreType.mongodb.value:
        from .mongodb import MongoDBKVStoreImpl

        impl = MongoDBKVStoreImpl(config)
    else:
        raise ValueError(f"Unknown kvstore type {config.type}")

    await impl.initialize()
    return impl
