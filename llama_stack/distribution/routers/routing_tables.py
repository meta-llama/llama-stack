# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403

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


# TODO: this routing table maintains state in memory purely. We need to
# add persistence to it when we add dynamic registration of objects.
class CommonRoutingTableImpl(RoutingTable):
    def __init__(
        self,
        registry: List[RoutableObject],
        impls_by_provider_id: Dict[str, RoutedProtocol],
    ) -> None:
        for obj in registry:
            if obj.provider_id not in impls_by_provider_id:
                raise ValueError(
                    f"Provider `{obj.provider_id}` pointed by `{obj.identifier}` not found"
                )

        self.impls_by_provider_id = impls_by_provider_id
        self.registry = registry

        for p in self.impls_by_provider_id.values():
            api = get_impl_api(p)
            if api == Api.inference:
                p.model_store = self
            elif api == Api.safety:
                p.shield_store = self
            elif api == Api.memory:
                p.memory_bank_store = self

        self.routing_key_to_object = {}
        for obj in self.registry:
            self.routing_key_to_object[obj.identifier] = obj

    async def initialize(self) -> None:
        for obj in self.registry:
            p = self.impls_by_provider_id[obj.provider_id]
            await register_object_with_provider(obj, p)

    async def shutdown(self) -> None:
        for p in self.impls_by_provider_id.values():
            await p.shutdown()

    def get_provider_impl(self, routing_key: str) -> Any:
        if routing_key not in self.routing_key_to_object:
            raise ValueError(f"Could not find provider for {routing_key}")
        obj = self.routing_key_to_object[routing_key]
        return self.impls_by_provider_id[obj.provider_id]

    def get_object_by_identifier(self, identifier: str) -> Optional[RoutableObject]:
        for obj in self.registry:
            if obj.identifier == identifier:
                return obj
        return None

    async def register_object(self, obj: RoutableObject) -> Any:
        if obj.identifier in self.routing_key_to_object:
            raise ValueError(f"Object `{obj.identifier}` already registered")

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]
        await register_object_with_provider(obj, p)

        self.routing_key_to_object[obj.identifier] = obj
        self.registry.append(obj)


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> List[ModelDef]:
        return self.registry

    async def get_model(self, identifier: str) -> Optional[ModelDef]:
        return self.get_object_by_identifier(identifier)

    async def register_model(self, model: ModelDef) -> None:
        await self.register_object(model)


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> List[ShieldDef]:
        return self.registry

    async def get_shield(self, shield_type: str) -> Optional[ShieldDef]:
        return self.get_object_by_identifier(shield_type)

    async def register_shield(self, shield: ShieldDef) -> None:
        await self.register_object(shield)


class MemoryBanksRoutingTable(CommonRoutingTableImpl, MemoryBanks):
    async def list_memory_banks(self) -> List[MemoryBankDef]:
        return self.registry

    async def get_memory_bank(self, identifier: str) -> Optional[MemoryBankDef]:
        return self.get_object_by_identifier(identifier)

    async def register_memory_bank(self, bank: MemoryBankDef) -> None:
        await self.register_object(bank)
