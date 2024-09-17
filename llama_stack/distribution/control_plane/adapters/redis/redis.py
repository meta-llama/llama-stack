# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime, timedelta
from typing import Any, List, Optional

from redis.asyncio import Redis

from llama_stack.apis.control_plane import *  # noqa: F403


from .config import RedisImplConfig


class RedisControlPlaneAdapter(ControlPlane):
    def __init__(self, config: RedisImplConfig):
        self.config = config

    async def initialize(self) -> None:
        self.redis = Redis.from_url(self.config.url)

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(
        self, key: str, value: Any, expiration: Optional[datetime] = None
    ) -> None:
        key = self._namespaced_key(key)
        await self.redis.set(key, value)
        if expiration:
            await self.redis.expireat(key, expiration)

    async def get(self, key: str) -> Optional[ControlPlaneValue]:
        key = self._namespaced_key(key)
        value = await self.redis.get(key)
        if value is None:
            return None
        ttl = await self.redis.ttl(key)
        expiration = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        return ControlPlaneValue(key=key, value=value, expiration=expiration)

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self.redis.delete(key)

    async def range(self, start_key: str, end_key: str) -> List[ControlPlaneValue]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        keys = await self.redis.keys(f"{start_key}*")
        result = []
        for key in keys:
            if key <= end_key:
                value = await self.get(key)
                if value:
                    result.append(value)
        return result
