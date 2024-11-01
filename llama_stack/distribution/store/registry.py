import json

from typing import Protocol

from docs.openapi_generator.strong_typing.deserializer import create_deserializer

from docs.openapi_generator.strong_typing.serialization import object_to_json

from llama_stack.distribution.datatypes import RoutableObjectWithProvider

from llama_stack.providers.utils.kvstore import KVStore


class Registry(Protocol):
    async def get(self, identifier: str) -> [RoutableObjectWithProvider]: ...
    async def register(self, obj: RoutableObjectWithProvider) -> None: ...


KEY_FORMAT = "distributions:registry:{}"
DESERIALIZER = create_deserializer(RoutableObjectWithProvider)


class DiskRegistry(Registry):
    def __init__(self, kvstore: KVStore):
        self.kvstore = kvstore

    async def get(self, identifier: str) -> [RoutableObjectWithProvider]:
        # Get JSON string from kvstore
        json_str = await self.kvstore.get(KEY_FORMAT.format(identifier))
        if not json_str:
            return []

        # Parse JSON string into list of objects
        objects_data = json.loads(json_str)

        return [DESERIALIZER.parse(obj_str) for obj_str in objects_data]

    # TODO: make it thread safe using CAS
    async def register(self, obj: RoutableObjectWithProvider) -> None:
        # Get existing objects for this identifier
        existing_objects = await self.get(obj.identifier)

        # Add new object to list
        existing_objects.append(obj)

        # Convert all objects to JSON strings and store as JSON array
        objects_json = [
            object_to_json(existing_object) for existing_object in existing_objects
        ]
        await self.kvstore.set(
            KEY_FORMAT.format(obj.identifier), json.dumps(objects_json)
        )
