# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import uuid
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter

from llama_stack.apis.benchmarks import Benchmark, Benchmarks, ListBenchmarksResponse
from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.datasets import (
    Dataset,
    DatasetPurpose,
    Datasets,
    DatasetType,
    DataSource,
    ListDatasetsResponse,
    RowsDataSource,
    URIDataSource,
)
from llama_stack.apis.models import ListModelsResponse, Model, Models, ModelType
from llama_stack.apis.resource import ResourceType
from llama_stack.apis.scoring_functions import (
    ListScoringFunctionsResponse,
    ScoringFn,
    ScoringFnParams,
    ScoringFunctions,
)
from llama_stack.apis.shields import ListShieldsResponse, Shield, Shields
from llama_stack.apis.tools import (
    ListToolGroupsResponse,
    ListToolsResponse,
    Tool,
    ToolGroup,
    ToolGroups,
    ToolHost,
)
from llama_stack.apis.vector_dbs import ListVectorDBsResponse, VectorDB, VectorDBs
from llama_stack.distribution.access_control import check_access
from llama_stack.distribution.datatypes import (
    AccessAttributes,
    BenchmarkWithACL,
    DatasetWithACL,
    ModelWithACL,
    RoutableObject,
    RoutableObjectWithProvider,
    RoutedProtocol,
    ScoringFnWithACL,
    ShieldWithACL,
    ToolGroupWithACL,
    ToolWithACL,
    VectorDBWithACL,
)
from llama_stack.distribution.request_headers import get_auth_attributes
from llama_stack.distribution.store import DistributionRegistry
from llama_stack.providers.datatypes import Api, RoutingTable

logger = logging.getLogger(__name__)


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
    elif api == Api.vector_io:
        return await p.register_vector_db(obj)
    elif api == Api.datasetio:
        return await p.register_dataset(obj)
    elif api == Api.scoring:
        return await p.register_scoring_function(obj)
    elif api == Api.eval:
        return await p.register_benchmark(obj)
    elif api == Api.tool_runtime:
        return await p.register_tool(obj)
    else:
        raise ValueError(f"Unknown API {api} for registering object with provider")


async def unregister_object_from_provider(obj: RoutableObject, p: Any) -> None:
    api = get_impl_api(p)
    if api == Api.vector_io:
        return await p.unregister_vector_db(obj.identifier)
    elif api == Api.inference:
        return await p.unregister_model(obj.identifier)
    elif api == Api.datasetio:
        return await p.unregister_dataset(obj.identifier)
    elif api == Api.tool_runtime:
        return await p.unregister_tool(obj.identifier)
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
        async def add_objects(objs: List[RoutableObjectWithProvider], provider_id: str, cls) -> None:
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
            elif api == Api.vector_io:
                p.vector_db_store = self
            elif api == Api.datasetio:
                p.dataset_store = self
            elif api == Api.scoring:
                p.scoring_function_store = self
                scoring_functions = await p.list_scoring_functions()
                await add_objects(scoring_functions, pid, ScoringFn)
            elif api == Api.eval:
                p.benchmark_store = self
            elif api == Api.tool_runtime:
                p.tool_store = self

    async def shutdown(self) -> None:
        for p in self.impls_by_provider_id.values():
            await p.shutdown()

    def get_provider_impl(self, routing_key: str, provider_id: Optional[str] = None) -> Any:
        def apiname_object():
            if isinstance(self, ModelsRoutingTable):
                return ("Inference", "model")
            elif isinstance(self, ShieldsRoutingTable):
                return ("Safety", "shield")
            elif isinstance(self, VectorDBsRoutingTable):
                return ("VectorIO", "vector_db")
            elif isinstance(self, DatasetsRoutingTable):
                return ("DatasetIO", "dataset")
            elif isinstance(self, ScoringFunctionsRoutingTable):
                return ("Scoring", "scoring_function")
            elif isinstance(self, BenchmarksRoutingTable):
                return ("Eval", "benchmark")
            elif isinstance(self, ToolGroupsRoutingTable):
                return ("Tools", "tool")
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

    async def get_object_by_identifier(self, type: str, identifier: str) -> Optional[RoutableObjectWithProvider]:
        # Get from disk registry
        obj = await self.dist_registry.get(type, identifier)
        if not obj:
            return None

        # Check if user has permission to access this object
        if not check_access(obj.identifier, getattr(obj, "access_attributes", None), get_auth_attributes()):
            logger.debug(f"Access denied to {type} '{identifier}' based on attribute mismatch")
            return None

        return obj

    async def unregister_object(self, obj: RoutableObjectWithProvider) -> None:
        await self.dist_registry.delete(obj.type, obj.identifier)
        await unregister_object_from_provider(obj, self.impls_by_provider_id[obj.provider_id])

    async def register_object(self, obj: RoutableObjectWithProvider) -> RoutableObjectWithProvider:
        # if provider_id is not specified, pick an arbitrary one from existing entries
        if not obj.provider_id and len(self.impls_by_provider_id) > 0:
            obj.provider_id = list(self.impls_by_provider_id.keys())[0]

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]

        # If object supports access control but no attributes set, use creator's attributes
        if not obj.access_attributes:
            creator_attributes = get_auth_attributes()
            if creator_attributes:
                obj.access_attributes = AccessAttributes(**creator_attributes)
                logger.info(f"Setting access attributes for {obj.type} '{obj.identifier}' based on creator's identity")

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
        filtered_objs = [obj for obj in objs if obj.type == type]

        # Apply attribute-based access control filtering
        if filtered_objs:
            filtered_objs = [
                obj
                for obj in filtered_objs
                if check_access(obj.identifier, getattr(obj, "access_attributes", None), get_auth_attributes())
            ]

        return filtered_objs


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    async def list_models(self) -> ListModelsResponse:
        return ListModelsResponse(data=await self.get_all_with_type("model"))

    async def get_model(self, model_id: str) -> Model:
        model = await self.get_object_by_identifier("model", model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        return model

    async def register_model(
        self,
        model_id: str,
        provider_model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_type: Optional[ModelType] = None,
    ) -> Model:
        if provider_model_id is None:
            provider_model_id = model_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this model
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    f"No provider specified and multiple providers available. Please specify a provider_id. Available providers: {self.impls_by_provider_id.keys()}"
                )
        if metadata is None:
            metadata = {}
        if model_type is None:
            model_type = ModelType.llm
        if "embedding_dimension" not in metadata and model_type == ModelType.embedding:
            raise ValueError("Embedding model must have an embedding dimension in its metadata")
        model = ModelWithACL(
            identifier=model_id,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(self, model_id: str) -> None:
        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ValueError(f"Model {model_id} not found")
        await self.unregister_object(existing_model)


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    async def list_shields(self) -> ListShieldsResponse:
        return ListShieldsResponse(data=await self.get_all_with_type(ResourceType.shield.value))

    async def get_shield(self, identifier: str) -> Shield:
        shield = await self.get_object_by_identifier("shield", identifier)
        if shield is None:
            raise ValueError(f"Shield '{identifier}' not found")
        return shield

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
        shield = ShieldWithACL(
            identifier=shield_id,
            provider_resource_id=provider_shield_id,
            provider_id=provider_id,
            params=params,
        )
        await self.register_object(shield)
        return shield


class VectorDBsRoutingTable(CommonRoutingTableImpl, VectorDBs):
    async def list_vector_dbs(self) -> ListVectorDBsResponse:
        return ListVectorDBsResponse(data=await self.get_all_with_type("vector_db"))

    async def get_vector_db(self, vector_db_id: str) -> VectorDB:
        vector_db = await self.get_object_by_identifier("vector_db", vector_db_id)
        if vector_db is None:
            raise ValueError(f"Vector DB '{vector_db_id}' not found")
        return vector_db

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: Optional[int] = 384,
        provider_id: Optional[str] = None,
        provider_vector_db_id: Optional[str] = None,
    ) -> VectorDB:
        if provider_vector_db_id is None:
            provider_vector_db_id = vector_db_id
        if provider_id is None:
            if len(self.impls_by_provider_id) > 0:
                provider_id = list(self.impls_by_provider_id.keys())[0]
                if len(self.impls_by_provider_id) > 1:
                    logger.warning(
                        f"No provider specified and multiple providers available. Arbitrarily selected the first provider {provider_id}."
                    )
            else:
                raise ValueError("No provider available. Please configure a vector_io provider.")
        model = await self.get_object_by_identifier("model", embedding_model)
        if model is None:
            raise ValueError(f"Model {embedding_model} not found")
        if model.model_type != ModelType.embedding:
            raise ValueError(f"Model {embedding_model} is not an embedding model")
        if "embedding_dimension" not in model.metadata:
            raise ValueError(f"Model {embedding_model} does not have an embedding dimension")
        vector_db_data = {
            "identifier": vector_db_id,
            "type": ResourceType.vector_db.value,
            "provider_id": provider_id,
            "provider_resource_id": provider_vector_db_id,
            "embedding_model": embedding_model,
            "embedding_dimension": model.metadata["embedding_dimension"],
        }
        vector_db = TypeAdapter(VectorDBWithACL).validate_python(vector_db_data)
        await self.register_object(vector_db)
        return vector_db

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        existing_vector_db = await self.get_vector_db(vector_db_id)
        if existing_vector_db is None:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        await self.unregister_object(existing_vector_db)


class DatasetsRoutingTable(CommonRoutingTableImpl, Datasets):
    async def list_datasets(self) -> ListDatasetsResponse:
        return ListDatasetsResponse(data=await self.get_all_with_type(ResourceType.dataset.value))

    async def get_dataset(self, dataset_id: str) -> Dataset:
        dataset = await self.get_object_by_identifier("dataset", dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        return dataset

    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dataset:
        if isinstance(source, dict):
            if source["type"] == "uri":
                source = URIDataSource.parse_obj(source)
            elif source["type"] == "rows":
                source = RowsDataSource.parse_obj(source)

        if not dataset_id:
            dataset_id = f"dataset-{str(uuid.uuid4())}"

        provider_dataset_id = dataset_id

        # infer provider from source
        if source.type == DatasetType.rows.value:
            provider_id = "localfs"
        elif source.type == DatasetType.uri.value:
            # infer provider from uri
            if source.uri.startswith("huggingface"):
                provider_id = "huggingface"
            else:
                provider_id = "localfs"
        else:
            raise ValueError(f"Unknown data source type: {source.type}")

        if metadata is None:
            metadata = {}

        dataset = DatasetWithACL(
            identifier=dataset_id,
            provider_resource_id=provider_dataset_id,
            provider_id=provider_id,
            purpose=purpose,
            source=source,
            metadata=metadata,
        )

        await self.register_object(dataset)
        return dataset

    async def unregister_dataset(self, dataset_id: str) -> None:
        dataset = await self.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        await self.unregister_object(dataset)


class ScoringFunctionsRoutingTable(CommonRoutingTableImpl, ScoringFunctions):
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse:
        return ListScoringFunctionsResponse(data=await self.get_all_with_type(ResourceType.scoring_function.value))

    async def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn:
        scoring_fn = await self.get_object_by_identifier("scoring_function", scoring_fn_id)
        if scoring_fn is None:
            raise ValueError(f"Scoring function '{scoring_fn_id}' not found")
        return scoring_fn

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
        scoring_fn = ScoringFnWithACL(
            identifier=scoring_fn_id,
            description=description,
            return_type=return_type,
            provider_resource_id=provider_scoring_fn_id,
            provider_id=provider_id,
            params=params,
        )
        scoring_fn.provider_id = provider_id
        await self.register_object(scoring_fn)


class BenchmarksRoutingTable(CommonRoutingTableImpl, Benchmarks):
    async def list_benchmarks(self) -> ListBenchmarksResponse:
        return ListBenchmarksResponse(data=await self.get_all_with_type("benchmark"))

    async def get_benchmark(self, benchmark_id: str) -> Benchmark:
        benchmark = await self.get_object_by_identifier("benchmark", benchmark_id)
        if benchmark is None:
            raise ValueError(f"Benchmark '{benchmark_id}' not found")
        return benchmark

    async def register_benchmark(
        self,
        benchmark_id: str,
        dataset_id: str,
        scoring_functions: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        provider_benchmark_id: Optional[str] = None,
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
        if provider_benchmark_id is None:
            provider_benchmark_id = benchmark_id
        benchmark = BenchmarkWithACL(
            identifier=benchmark_id,
            dataset_id=dataset_id,
            scoring_functions=scoring_functions,
            metadata=metadata,
            provider_id=provider_id,
            provider_resource_id=provider_benchmark_id,
        )
        await self.register_object(benchmark)


class ToolGroupsRoutingTable(CommonRoutingTableImpl, ToolGroups):
    async def list_tools(self, toolgroup_id: Optional[str] = None) -> ListToolsResponse:
        tools = await self.get_all_with_type("tool")
        if toolgroup_id:
            tools = [tool for tool in tools if tool.toolgroup_id == toolgroup_id]
        return ListToolsResponse(data=tools)

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        return ListToolGroupsResponse(data=await self.get_all_with_type("tool_group"))

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup:
        tool_group = await self.get_object_by_identifier("tool_group", toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group '{toolgroup_id}' not found")
        return tool_group

    async def get_tool(self, tool_name: str) -> Tool:
        return await self.get_object_by_identifier("tool", tool_name)

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: Optional[URL] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        tools = []
        tool_defs = await self.impls_by_provider_id[provider_id].list_runtime_tools(toolgroup_id, mcp_endpoint)
        tool_host = ToolHost.model_context_protocol if mcp_endpoint else ToolHost.distribution

        for tool_def in tool_defs:
            tools.append(
                ToolWithACL(
                    identifier=tool_def.name,
                    toolgroup_id=toolgroup_id,
                    description=tool_def.description or "",
                    parameters=tool_def.parameters or [],
                    provider_id=provider_id,
                    provider_resource_id=tool_def.name,
                    metadata=tool_def.metadata,
                    tool_host=tool_host,
                )
            )
        for tool in tools:
            existing_tool = await self.get_tool(tool.identifier)
            # Compare existing and new object if one exists
            if existing_tool:
                existing_dict = existing_tool.model_dump()
                new_dict = tool.model_dump()

                if existing_dict != new_dict:
                    raise ValueError(
                        f"Object {tool.identifier} already exists in registry. Please use a different identifier."
                    )
            await self.register_object(tool)

        await self.dist_registry.register(
            ToolGroupWithACL(
                identifier=toolgroup_id,
                provider_id=provider_id,
                provider_resource_id=toolgroup_id,
                mcp_endpoint=mcp_endpoint,
                args=args,
            )
        )

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        tool_group = await self.get_tool_group(toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group {toolgroup_id} not found")
        tools = (await self.list_tools(toolgroup_id)).data
        for tool in tools:
            await self.unregister_object(tool)
        await self.unregister_object(tool_group)

    async def shutdown(self) -> None:
        pass
