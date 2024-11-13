# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Protocol, Tuple

import pydantic

from llama_stack.distribution.datatypes import KVStoreConfig, RoutableObjectWithProvider
from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR

from llama_stack.providers.utils.kvstore import (
    KVStore,
    kvstore_impl,
    SqliteKVStoreConfig,
)


class DistributionRegistry(Protocol):
    async def get_all(self) -> List[RoutableObjectWithProvider]: ...

    async def initialize(self) -> None: ...

    async def get(self, identifier: str) -> List[RoutableObjectWithProvider]: ...

    def get_cached(self, identifier: str) -> List[RoutableObjectWithProvider]: ...

    # The current data structure allows multiple objects with the same identifier but different providers.
    # This is not ideal - we should have a single object that can be served by multiple providers,
    # suggesting a data structure like (obj: Obj, providers: List[str]) rather than List[RoutableObjectWithProvider].
    # The current approach could lead to inconsistencies if the same logical object has different data across providers.
    async def register(self, obj: RoutableObjectWithProvider) -> bool: ...


REGISTER_PREFIX = "distributions:registry"
KEY_VERSION = "v1"
KEY_FORMAT = f"{REGISTER_PREFIX}:{KEY_VERSION}::" + "{type}:{identifier}"


def _get_registry_key_range() -> Tuple[str, str]:
    """Returns the start and end keys for the registry range query."""
    start_key = f"{REGISTER_PREFIX}:{KEY_VERSION}"
    return start_key, f"{start_key}\xff"


def _parse_registry_values(values: List[str]) -> List[RoutableObjectWithProvider]:
    """Utility function to parse registry values into RoutableObjectWithProvider objects."""
    all_objects = []
    for value in values:
        try:
            objects_data = json.loads(value)
            objects = [
                pydantic.parse_obj_as(
                    RoutableObjectWithProvider,
                    json.loads(obj_str),
                )
                for obj_str in objects_data
            ]
            all_objects.extend(objects)
        except Exception as e:
            print(f"Error parsing value: {e}")
            traceback.print_exc()
    return all_objects


class DiskDistributionRegistry(DistributionRegistry):
    def __init__(self, kvstore: KVStore):
        self.kvstore = kvstore

    async def initialize(self) -> None:
        pass

    def get_cached(
        self, type: str, identifier: str
    ) -> List[RoutableObjectWithProvider]:
        # Disk registry does not have a cache
        return []

    async def get_all(self) -> List[RoutableObjectWithProvider]:
        start_key, end_key = _get_registry_key_range()
        values = await self.kvstore.range(start_key, end_key)
        return _parse_registry_values(values)

    async def get(self, type: str, identifier: str) -> List[RoutableObjectWithProvider]:
        json_str = await self.kvstore.get(
            KEY_FORMAT.format(type=type, identifier=identifier)
        )
        if not json_str:
            return []

        objects_data = json.loads(json_str)
        return [
            pydantic.parse_obj_as(
                RoutableObjectWithProvider,
                json.loads(obj_str),
            )
            for obj_str in objects_data
        ]

    async def register(self, obj: RoutableObjectWithProvider) -> bool:
        existing_objects = await self.get(obj.type, obj.identifier)
        # dont register if the object's providerid already exists
        for eobj in existing_objects:
            if eobj.provider_id == obj.provider_id:
                return False

        existing_objects.append(obj)

        objects_json = [
            obj.model_dump_json() for obj in existing_objects
        ]  # Fixed variable name
        await self.kvstore.set(
            KEY_FORMAT.format(type=obj.type, identifier=obj.identifier),
            json.dumps(objects_json),
        )
        return True


class CachedDiskDistributionRegistry(DiskDistributionRegistry):
    def __init__(self, kvstore: KVStore):
        super().__init__(kvstore)
        self.cache: Dict[Tuple[str, str], List[RoutableObjectWithProvider]] = {}
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()

    @asynccontextmanager
    async def _locked_cache(self):
        """Context manager for safely accessing the cache with a lock."""
        async with self._cache_lock:
            yield self.cache

    async def _ensure_initialized(self):
        """Ensures the registry is initialized before operations."""
        if self._initialized:
            return

        async with self._initialize_lock:
            if self._initialized:
                return

            start_key, end_key = _get_registry_key_range()
            values = await self.kvstore.range(start_key, end_key)
            objects = _parse_registry_values(values)

            async with self._locked_cache() as cache:
                for obj in objects:
                    cache_key = (obj.type, obj.identifier)
                    if cache_key not in cache:
                        cache[cache_key] = []
                    if not any(
                        cached_obj.provider_id == obj.provider_id
                        for cached_obj in cache[cache_key]
                    ):
                        cache[cache_key].append(obj)

            self._initialized = True

    async def initialize(self) -> None:
        await self._ensure_initialized()

    def get_cached(
        self, type: str, identifier: str
    ) -> List[RoutableObjectWithProvider]:
        return self.cache.get((type, identifier), [])[:]  # Return a copy

    async def get_all(self) -> List[RoutableObjectWithProvider]:
        await self._ensure_initialized()
        async with self._locked_cache() as cache:
            return [item for sublist in cache.values() for item in sublist]

    async def get(self, type: str, identifier: str) -> List[RoutableObjectWithProvider]:
        await self._ensure_initialized()
        cache_key = (type, identifier)

        async with self._locked_cache() as cache:
            if cache_key in cache:
                return cache[cache_key][:]

        objects = await super().get(type, identifier)
        if objects:
            async with self._locked_cache() as cache:
                cache[cache_key] = objects

        return objects

    async def register(self, obj: RoutableObjectWithProvider) -> bool:
        await self._ensure_initialized()
        success = await super().register(obj)

        if success:
            cache_key = (obj.type, obj.identifier)
            async with self._locked_cache() as cache:
                if cache_key not in cache:
                    cache[cache_key] = []
                if not any(
                    cached_obj.provider_id == obj.provider_id
                    for cached_obj in cache[cache_key]
                ):
                    cache[cache_key].append(obj)

        return success


async def create_dist_registry(
    metadata_store: Optional[KVStoreConfig],
    image_name: str,
) -> tuple[CachedDiskDistributionRegistry, KVStore]:
    # instantiate kvstore for storing and retrieving distribution metadata
    if metadata_store:
        dist_kvstore = await kvstore_impl(metadata_store)
    else:
        dist_kvstore = await kvstore_impl(
            SqliteKVStoreConfig(
                db_path=(DISTRIBS_BASE_DIR / image_name / "kvstore.db").as_posix()
            )
        )

    return CachedDiskDistributionRegistry(dist_kvstore), dist_kvstore
