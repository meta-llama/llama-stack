# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403

from llama_stack.distribution.datatypes import *  # noqa: F403


def get_impl_api(p: Any) -> Api:
    return p.__provider_spec__.api


async def register_object_with_provider(obj: RoutableObject, p: Any) -> None:
    api = get_impl_api(p)
    if api == Api.inference:
        await p.register_model(obj)
    elif api == Api.safety:
        await p.register_shield(obj)
    elif api == Api.memory:
        await p.register_memory_bank(obj)
    elif api == Api.datasetio:
        await p.register_dataset(obj)
    else:
        raise ValueError(f"Unknown API {api} for registering object with provider")


Registry = Dict[str, List[RoutableObjectWithProvider]]


# TODO: this routing table maintains state in memory purely. We need to
# add persistence to it when we add dynamic registration of objects.
class CommonRoutingTableImpl(RoutingTable):
    def __init__(
        self,
        impls_by_provider_id: Dict[str, RoutedProtocol],
    ) -> None:
        self.impls_by_provider_id = impls_by_provider_id

    async def initialize(self) -> None:
        self.registry: Registry = {}

        def add_objects(objs: List[RoutableObjectWithProvider]) -> None:
            for obj in objs:
                if obj.identifier not in self.registry:
                    self.registry[obj.identifier] = []

                self.registry[obj.identifier].append(obj)

        for pid, p in self.impls_by_provider_id.items():
            api = get_impl_api(p)
            if api == Api.inference:
                p.model_store = self
                models = await p.list_models()
                add_objects(
                    [ModelDefWithProvider(**m.dict(), provider_id=pid) for m in models]
                )

            elif api == Api.safety:
                p.shield_store = self
                shields = await p.list_shields()
                add_objects(
                    [
                        ShieldDefWithProvider(**s.dict(), provider_id=pid)
                        for s in shields
                    ]
                )

            elif api == Api.memory:
                p.memory_bank_store = self
                memory_banks = await p.list_memory_banks()

                # do in-memory updates due to pesky Annotated unions
                for m in memory_banks:
                    m.provider_id = pid

                add_objects(memory_banks)

            elif api == Api.datasetio:
                p.dataset_store = self
                datasets = await p.list_datasets()

                # do in-memory updates due to pesky Annotated unions
                for d in datasets:
                    d.provider_id = pid

                add_objects(datasets)

    async def shutdown(self) -> None:
        for p in self.impls_by_provider_id.values():
            await p.shutdown()

    def get_provider_impl(
        self, routing_key: str, provider_id: Optional[str] = None
    ) -> Any:
        def apiname_object():
            if isinstance(self, ModelsRoutingTable):
                return ("Inference", "model")
            elif isinstance(self, ShieldsRoutingTable):
                return ("Safety", "shield")
            elif isinstance(self, MemoryBanksRoutingTable):
                return ("Memory", "memory_bank")
            else:
                raise ValueError("Unknown routing table type")

        if routing_key not in self.registry:
            apiname, objname = apiname_object()
            raise ValueError(
                f"`{routing_key}` not registered. Make sure there is an {apiname} provider serving this {objname}."
            )

        objs = self.registry[routing_key]
        for obj in objs:
            if not provider_id or provider_id == obj.provider_id:
                return self.impls_by_provider_id[obj.provider_id]

        raise ValueError(f"Provider not found for `{routing_key}`")

    def get_object_by_identifier(
        self, identifier: str
    ) -> Optional[RoutableObjectWithProvider]:
        objs = self.registry.get(identifier, [])
        if not objs:
            return None

        # kind of ill-defined behavior here, but we'll just return the first one
        return objs[0]

    async def register_object(self, obj: RoutableObjectWithProvider):
        entries = self.registry.get(obj.identifier, [])
        for entry in entries:
            if entry.provider_id == obj.provider_id or not obj.provider_id:
                print(
                    f"`{obj.identifier}` already registered with `{entry.provider_id}`"
                )
                return

        # if provider_id is not specified, we'll pick an arbitrary one from existing entries
        if not obj.provider_id and len(self.impls_by_provider_id) > 0:
            obj.provider_id = list(self.impls_by_provider_id.keys())[0]

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]

        await register_object_with_provider(obj, p)

        if obj.identifier not in self.registry:
            self.registry[obj.identifier] = []
        self.registry[obj.identifier].append(obj)

        # TODO: persist this to a store


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> List[ModelDefWithProvider]:
        objects = []
        for objs in self.registry.values():
            objects.extend(objs)
        return objects

    async def get_model(self, identifier: str) -> Optional[ModelDefWithProvider]:
        return self.get_object_by_identifier(identifier)

    async def register_model(self, model: ModelDefWithProvider) -> None:
        await self.register_object(model)


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> List[ShieldDef]:
        objects = []
        for objs in self.registry.values():
            objects.extend(objs)
        return objects

    async def get_shield(self, shield_type: str) -> Optional[ShieldDefWithProvider]:
        return self.get_object_by_identifier(shield_type)

    async def register_shield(self, shield: ShieldDefWithProvider) -> None:
        await self.register_object(shield)


class MemoryBanksRoutingTable(CommonRoutingTableImpl, MemoryBanks):
    async def list_memory_banks(self) -> List[MemoryBankDefWithProvider]:
        objects = []
        for objs in self.registry.values():
            objects.extend(objs)
        return objects

    async def get_memory_bank(
        self, identifier: str
    ) -> Optional[MemoryBankDefWithProvider]:
        return self.get_object_by_identifier(identifier)

    async def register_memory_bank(
        self, memory_bank: MemoryBankDefWithProvider
    ) -> None:
        await self.register_object(memory_bank)


class DatasetsRoutingTable(CommonRoutingTableImpl, Datasets):
    async def list_datasets(self) -> List[DatasetDefWithProvider]:
        objects = []
        for objs in self.registry.values():
            objects.extend(objs)
        return objects

    async def get_dataset(
        self, dataset_identifier: str
    ) -> Optional[DatasetDefWithProvider]:
        return self.get_object_by_identifier(identifier)

    async def register_dataset(self, dataset_def: DatasetDefWithProvider) -> None:
        await self.register_object(dataset_def)
