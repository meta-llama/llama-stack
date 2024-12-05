# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from pydantic import parse_obj_as

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.eval_tasks import *  # noqa: F403


from llama_models.llama3.api.datatypes import URL

from llama_stack.apis.common.type_system import ParamType
from llama_stack.distribution.store import DistributionRegistry
from llama_stack.distribution.datatypes import *  # noqa: F403


def get_impl_api(p: Any) -> Api:
    return p.__provider_spec__.api


# TODO: this should return the registered object for all APIs
async def register_object_with_provider(obj: RoutableObject, p: Any) -> RoutableObject:

    api = get_impl_api(p)

    assert obj.provider_id != "remote", "Remote provider should not be registered"

    if api == Api.inference:
        return await p.register_model(obj)
    elif api == Api.safety:
        return await p.register_shield(obj)
    elif api == Api.memory:
        return await p.register_memory_bank(obj)
    elif api == Api.datasetio:
        return await p.register_dataset(obj)
    elif api == Api.scoring:
        return await p.register_scoring_function(obj)
    elif api == Api.eval:
        return await p.register_eval_task(obj)
    else:
        raise ValueError(f"Unknown API {api} for registering object with provider")


async def unregister_object_from_provider(obj: RoutableObject, p: Any) -> None:
    api = get_impl_api(p)
    if api == Api.memory:
        return await p.unregister_memory_bank(obj.identifier)
    elif api == Api.inference:
        return await p.unregister_model(obj.identifier)
    elif api == Api.datasetio:
        return await p.unregister_dataset(obj.identifier)
    else:
        raise ValueError(f"Unregister not supported for {api}")


Registry = Dict[str, List[RoutableObjectWithProvider]]


class CommonRoutingTableImpl(RoutingTable):
    def __init__(
        self,
        impls_by_provider_id: Dict[str, RoutedProtocol],
        dist_registry: DistributionRegistry,
    ) -> None:
        self.impls_by_provider_id = impls_by_provider_id
        self.dist_registry = dist_registry

    async def initialize(self) -> None:

        async def add_objects(
            objs: List[RoutableObjectWithProvider], provider_id: str, cls
        ) -> None:
            for obj in objs:
                if cls is None:
                    obj.provider_id = provider_id
                else:
                    # Create a copy of the model data and explicitly set provider_id
                    model_data = obj.model_dump()
                    model_data["provider_id"] = provider_id
                    obj = cls(**model_data)
                await self.dist_registry.register(obj)

        # Register all objects from providers
        for pid, p in self.impls_by_provider_id.items():
            api = get_impl_api(p)
            if api == Api.inference:
                p.model_store = self
            elif api == Api.safety:
                p.shield_store = self
            elif api == Api.memory:
                p.memory_bank_store = self
            elif api == Api.datasetio:
                p.dataset_store = self
            elif api == Api.scoring:
                p.scoring_function_store = self
                scoring_functions = await p.list_scoring_functions()
                await add_objects(scoring_functions, pid, ScoringFn)
            elif api == Api.eval:
                p.eval_task_store = self

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
            elif isinstance(self, DatasetsRoutingTable):
                return ("DatasetIO", "dataset")
            elif isinstance(self, ScoringFunctionsRoutingTable):
                return ("Scoring", "scoring_function")
            elif isinstance(self, EvalTasksRoutingTable):
                return ("Eval", "eval_task")
            else:
                raise ValueError("Unknown routing table type")

        apiname, objtype = apiname_object()

        # Get objects from disk registry
        obj = self.dist_registry.get_cached(objtype, routing_key)
        if not obj:
            provider_ids = list(self.impls_by_provider_id.keys())
            if len(provider_ids) > 1:
                provider_ids_str = f"any of the providers: {', '.join(provider_ids)}"
            else:
                provider_ids_str = f"provider: `{provider_ids[0]}`"
            raise ValueError(
                f"{objtype.capitalize()} `{routing_key}` not served by {provider_ids_str}. Make sure there is an {apiname} provider serving this {objtype}."
            )

        if not provider_id or provider_id == obj.provider_id:
            return self.impls_by_provider_id[obj.provider_id]

        raise ValueError(f"Provider not found for `{routing_key}`")

    async def get_object_by_identifier(
        self, type: str, identifier: str
    ) -> Optional[RoutableObjectWithProvider]:
        # Get from disk registry
        obj = await self.dist_registry.get(type, identifier)
        if not obj:
            return None

        return obj

    async def unregister_object(self, obj: RoutableObjectWithProvider) -> None:
        await self.dist_registry.delete(obj.type, obj.identifier)
        await unregister_object_from_provider(
            obj, self.impls_by_provider_id[obj.provider_id]
        )

    async def register_object(
        self, obj: RoutableObjectWithProvider
    ) -> RoutableObjectWithProvider:
        # Get existing objects from registry
        existing_obj = await self.dist_registry.get(obj.type, obj.identifier)

        # if provider_id is not specified, pick an arbitrary one from existing entries
        if not obj.provider_id and len(self.impls_by_provider_id) > 0:
            obj.provider_id = list(self.impls_by_provider_id.keys())[0]

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]

        registered_obj = await register_object_with_provider(obj, p)
        # TODO: This needs to be fixed for all APIs once they return the registered object
        if obj.type == ResourceType.model.value:
            await self.dist_registry.register(registered_obj)
            return registered_obj

        else:
            await self.dist_registry.register(obj)
            return obj

    async def get_all_with_type(self, type: str) -> List[RoutableObjectWithProvider]:
        objs = await self.dist_registry.get_all()
        return [obj for obj in objs if obj.type == type]


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> List[Model]:
        return await self.get_all_with_type("model")

    async def get_model(self, identifier: str) -> Optional[Model]:
        return await self.get_object_by_identifier("model", identifier)

    async def register_model(
        self,
        model_id: str,
        provider_model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Model:
        if provider_model_id is None:
            provider_model_id = model_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this model
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id. Available providers: {self.impls_by_provider_id.keys()}"
                )
        if metadata is None:
            metadata = {}
        model = Model(
            identifier=model_id,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(self, model_id: str) -> None:
        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ValueError(f"Model {model_id} not found")
        await self.unregister_object(existing_model)


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> List[Shield]:
        return await self.get_all_with_type(ResourceType.shield.value)

    async def get_shield(self, identifier: str) -> Optional[Shield]:
        return await self.get_object_by_identifier("shield", identifier)

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield:
        if provider_shield_id is None:
            provider_shield_id = shield_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this shield type
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        if params is None:
            params = {}
        shield = Shield(
            identifier=shield_id,
            provider_resource_id=provider_shield_id,
            provider_id=provider_id,
            params=params,
        )
        await self.register_object(shield)
        return shield


class MemoryBanksRoutingTable(CommonRoutingTableImpl, MemoryBanks):
    async def list_memory_banks(self) -> List[MemoryBank]:
        return await self.get_all_with_type(ResourceType.memory_bank.value)

    async def get_memory_bank(self, memory_bank_id: str) -> Optional[MemoryBank]:
        return await self.get_object_by_identifier("memory_bank", memory_bank_id)

    async def register_memory_bank(
        self,
        memory_bank_id: str,
        params: BankParams,
        provider_id: Optional[str] = None,
        provider_memory_bank_id: Optional[str] = None,
    ) -> MemoryBank:
        if provider_memory_bank_id is None:
            provider_memory_bank_id = memory_bank_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this shield type
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        memory_bank = parse_obj_as(
            MemoryBank,
            {
                "identifier": memory_bank_id,
                "type": ResourceType.memory_bank.value,
                "provider_id": provider_id,
                "provider_resource_id": provider_memory_bank_id,
                **params.model_dump(),
            },
        )
        await self.register_object(memory_bank)
        return memory_bank

    async def unregister_memory_bank(self, memory_bank_id: str) -> None:
        existing_bank = await self.get_memory_bank(memory_bank_id)
        if existing_bank is None:
            raise ValueError(f"Memory bank {memory_bank_id} not found")
        await self.unregister_object(existing_bank)


class DatasetsRoutingTable(CommonRoutingTableImpl, Datasets):
    async def list_datasets(self) -> List[Dataset]:
        return await self.get_all_with_type(ResourceType.dataset.value)

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        return await self.get_object_by_identifier("dataset", dataset_id)

    async def register_dataset(
        self,
        dataset_id: str,
        dataset_schema: Dict[str, ParamType],
        url: URL,
        provider_dataset_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if provider_dataset_id is None:
            provider_dataset_id = dataset_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this dataset
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        if metadata is None:
            metadata = {}
        dataset = Dataset(
            identifier=dataset_id,
            provider_resource_id=provider_dataset_id,
            provider_id=provider_id,
            dataset_schema=dataset_schema,
            url=url,
            metadata=metadata,
        )
        await self.register_object(dataset)

    async def unregister_dataset(self, dataset_id: str) -> None:
        dataset = await self.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        await self.unregister_object(dataset)


class ScoringFunctionsRoutingTable(CommonRoutingTableImpl, ScoringFunctions):
    async def list_scoring_functions(self) -> List[ScoringFn]:
        return await self.get_all_with_type(ResourceType.scoring_function.value)

    async def get_scoring_function(self, scoring_fn_id: str) -> Optional[ScoringFn]:
        return await self.get_object_by_identifier("scoring_function", scoring_fn_id)

    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[ScoringFnParams] = None,
    ) -> None:
        if provider_scoring_fn_id is None:
            provider_scoring_fn_id = scoring_fn_id
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        scoring_fn = ScoringFn(
            identifier=scoring_fn_id,
            description=description,
            return_type=return_type,
            provider_resource_id=provider_scoring_fn_id,
            provider_id=provider_id,
            params=params,
        )
        scoring_fn.provider_id = provider_id
        await self.register_object(scoring_fn)


class EvalTasksRoutingTable(CommonRoutingTableImpl, EvalTasks):
    async def list_eval_tasks(self) -> List[EvalTask]:
        return await self.get_all_with_type(ResourceType.eval_task.value)

    async def get_eval_task(self, name: str) -> Optional[EvalTask]:
        return await self.get_object_by_identifier("eval_task", name)

    async def register_eval_task(
        self,
        eval_task_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        provider_eval_task_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        if provider_eval_task_id is None:
            provider_eval_task_id = eval_task_id
        eval_task = EvalTask(
            identifier=eval_task_id,
            dataset_id=dataset_id,
            scoring_functions=scoring_functions,
            metadata=metadata,
            provider_id=provider_id,
            provider_resource_id=provider_eval_task_id,
        )
        await self.register_object(eval_task)
