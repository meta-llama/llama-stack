# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Optional, Union

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.inference import Inference
from llama_stack.apis.memory import Memory
from llama_stack.apis.safety import Safety

from llama_stack.distribution.datatypes import *  # noqa: F403


RoutableObject = Union[
    ModelDef,
    ShieldDef,
    MemoryBankDef,
]

RoutedProtocol = Union[
    Inference,
    Safety,
    Memory,
]


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

    async def initialize(self) -> None:
        keys_by_provider = {}
        for obj in self.registry:
            keys = keys_by_provider.setdefault(obj.provider_id, [])
            keys.append(obj.routing_key)

        for provider_id, keys in keys_by_provider.items():
            p = self.impls_by_provider_id[provider_id]
            spec = p.__provider_spec__
            if is_passthrough(spec):
                continue

            await p.validate_routing_keys(keys)

    async def shutdown(self) -> None:
        pass

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


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> List[ModelDef]:
        return self.registry

    async def get_model(self, identifier: str) -> Optional[ModelDef]:
        return self.get_object_by_identifier(identifier)

    async def register_model(self, model: ModelDef) -> None:
        raise NotImplementedError()


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> List[ShieldDef]:
        return self.registry

    async def get_shield(self, shield_type: str) -> Optional[ShieldDef]:
        return self.get_object_by_identifier(shield_type)

    async def register_shield(self, shield: ShieldDef) -> None:
        raise NotImplementedError()


class MemoryBanksRoutingTable(CommonRoutingTableImpl, MemoryBanks):
    async def list_memory_banks(self) -> List[MemoryBankDef]:
        return self.registry

    async def get_memory_bank(self, identifier: str) -> Optional[MemoryBankDef]:
        return self.get_object_by_identifier(identifier)

    async def register_memory_bank(self, bank: MemoryBankDef) -> None:
        raise NotImplementedError()
