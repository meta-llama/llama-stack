# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict, List, Protocol

import pydantic

from llama_stack.distribution.datatypes import RoutableObjectWithProvider

from llama_stack.providers.utils.kvstore import KVStore


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


KEY_FORMAT = "distributions:registry:{}"


class DiskDistributionRegistry(DistributionRegistry):
    def __init__(self, kvstore: KVStore):
        self.kvstore = kvstore

    async def initialize(self) -> None:
        pass

    def get_cached(self, identifier: str) -> List[RoutableObjectWithProvider]:
        # Disk registry does not have a cache
        return []

    async def get_all(self) -> List[RoutableObjectWithProvider]:
        start_key = KEY_FORMAT.format("")
        end_key = KEY_FORMAT.format("\xff")
        keys = await self.kvstore.range(start_key, end_key)
        return [await self.get(key.split(":")[-1]) for key in keys]

    async def get(self, identifier: str) -> List[RoutableObjectWithProvider]:
        json_str = await self.kvstore.get(KEY_FORMAT.format(identifier))
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
        existing_objects = await self.get(obj.identifier)
        # dont register if the object's providerid already exists
        for eobj in existing_objects:
            if eobj.provider_id == obj.provider_id:
                return False

        existing_objects.append(obj)

        objects_json = [
            obj.model_dump_json() for obj in existing_objects
        ]  # Fixed variable name
        await self.kvstore.set(
            KEY_FORMAT.format(obj.identifier), json.dumps(objects_json)
        )
        return True


class CachedDiskDistributionRegistry(DiskDistributionRegistry):
    def __init__(self, kvstore: KVStore):
        super().__init__(kvstore)
        self.cache: Dict[str, List[RoutableObjectWithProvider]] = {}

    async def initialize(self) -> None:
        start_key = KEY_FORMAT.format("")
        end_key = KEY_FORMAT.format("\xff")

        keys = await self.kvstore.range(start_key, end_key)

        for key in keys:
            identifier = key.split(":")[-1]
            objects = await super().get(identifier)
            if objects:
                self.cache[identifier] = objects

    def get_cached(self, identifier: str) -> List[RoutableObjectWithProvider]:
        return self.cache.get(identifier, [])

    async def get_all(self) -> List[RoutableObjectWithProvider]:
        return [item for sublist in self.cache.values() for item in sublist]

    async def get(self, identifier: str) -> List[RoutableObjectWithProvider]:
        if identifier in self.cache:
            return self.cache[identifier]

        objects = await super().get(identifier)
        if objects:
            self.cache[identifier] = objects

        return objects

    async def register(self, obj: RoutableObjectWithProvider) -> bool:
        # First update disk
        success = await super().register(obj)

        if success:
            # Then update cache
            if obj.identifier not in self.cache:
                self.cache[obj.identifier] = []

            # Check if provider already exists in cache
            for cached_obj in self.cache[obj.identifier]:
                if cached_obj.provider_id == obj.provider_id:
                    return success

            # If not, update cache
            self.cache[obj.identifier].append(obj)

        return success
