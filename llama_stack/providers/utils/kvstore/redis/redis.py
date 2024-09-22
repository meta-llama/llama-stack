# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime, timedelta
from typing import List, Optional

from redis.asyncio import Redis

from ..api import *  # noqa: F403
from ..config import RedisKVStoreImplConfig


class RedisKVStoreImpl(KVStore):
    def __init__(self, config: RedisKVStoreImplConfig):
        self.config = config

    async def initialize(self) -> None:
        self.redis = Redis.from_url(self.config.url)

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(
        self, key: str, value: str, expiration: Optional[datetime] = None
    ) -> None:
        key = self._namespaced_key(key)
        await self.redis.set(key, value)
        if expiration:
            await self.redis.expireat(key, expiration)

    async def get(self, key: str) -> Optional[KVStoreValue]:
        key = self._namespaced_key(key)
        value = await self.redis.get(key)
        if value is None:
            return None
        ttl = await self.redis.ttl(key)
        expiration = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        return KVStoreValue(key=key, value=value, expiration=expiration)

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self.redis.delete(key)

    async def range(self, start_key: str, end_key: str) -> List[KVStoreValue]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        return await self.redis.zrangebylex(start_key, end_key)
