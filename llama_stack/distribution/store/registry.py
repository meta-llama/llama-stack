# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from typing import Protocol

import pydantic

from llama_stack.distribution.datatypes import RoutableObjectWithProvider

from llama_stack.providers.utils.kvstore import KVStore


class DistributionRegistry(Protocol):
    
    async def get(self, identifier: str) -> [RoutableObjectWithProvider]: ...
    # The current data structure allows multiple objects with the same identifier but different providers.
    # This is not ideal - we should have a single object that can be served by multiple providers,
    # suggesting a data structure like (obj: Obj, providers: List[str]) rather than List[RoutableObjectWithProvider].
    # The current approach could lead to inconsistencies if the same logical object has different data across providers.
    async def register(self, obj: RoutableObjectWithProvider) -> None: ...


KEY_FORMAT = "distributions:registry:{}"


class DiskDistributionRegistry(DistributionRegistry):
    def __init__(self, kvstore: KVStore):
        self.kvstore = kvstore

    async def get(self, identifier: str) -> [RoutableObjectWithProvider]:
        # Get JSON string from kvstore
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

    # TODO: make it thread safe using CAS
    async def register(self, obj: RoutableObjectWithProvider) -> None:
        existing_objects = await self.get(obj.identifier)
        # dont register if the object's providerid already exists
        for eobj in existing_objects:
            if eobj.provider_id == obj.provider_id:
                return

        existing_objects.append(obj)

        objects_json = [obj.model_dump_json() for existing_object in existing_objects]
        await self.kvstore.set(
            KEY_FORMAT.format(obj.identifier), json.dumps(objects_json)
        )
