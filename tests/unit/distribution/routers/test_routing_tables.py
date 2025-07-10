# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Unit tests for the routing tables

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.datasets.datasets import Dataset, DatasetPurpose, URIDataSource
from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import Model, ModelType
from llama_stack.apis.shields.shields import Shield
from llama_stack.apis.tools import ListToolDefsResponse, ToolDef, ToolGroup, ToolParameter
from llama_stack.apis.vector_dbs.vector_dbs import VectorDB
from llama_stack.distribution.routing_tables.benchmarks import BenchmarksRoutingTable
from llama_stack.distribution.routing_tables.datasets import DatasetsRoutingTable
from llama_stack.distribution.routing_tables.models import ModelsRoutingTable
from llama_stack.distribution.routing_tables.scoring_functions import ScoringFunctionsRoutingTable
from llama_stack.distribution.routing_tables.shields import ShieldsRoutingTable
from llama_stack.distribution.routing_tables.toolgroups import ToolGroupsRoutingTable
from llama_stack.distribution.routing_tables.vector_dbs import VectorDBsRoutingTable


class Impl:
    def __init__(self, api: Api):
        self.api = api

    @property
    def __provider_spec__(self):
        _provider_spec = AsyncMock()
        _provider_spec.api = self.api
        return _provider_spec


class InferenceImpl(Impl):
    def __init__(self):
        super().__init__(Api.inference)

    async def register_model(self, model: Model):
        return model

    async def unregister_model(self, model_id: str):
        return model_id


class SafetyImpl(Impl):
    def __init__(self):
        super().__init__(Api.safety)

    async def register_shield(self, shield: Shield):
        return shield


class VectorDBImpl(Impl):
    def __init__(self):
        super().__init__(Api.vector_io)

    async def register_vector_db(self, vector_db: VectorDB):
        return vector_db

    async def unregister_vector_db(self, vector_db_id: str):
        return vector_db_id


class DatasetsImpl(Impl):
    def __init__(self):
        super().__init__(Api.datasetio)

    async def register_dataset(self, dataset: Dataset):
        return dataset

    async def unregister_dataset(self, dataset_id: str):
        return dataset_id


class ScoringFunctionsImpl(Impl):
    def __init__(self):
        super().__init__(Api.scoring)

    async def list_scoring_functions(self):
        return []

    async def register_scoring_function(self, scoring_fn):
        return scoring_fn


class BenchmarksImpl(Impl):
    def __init__(self):
        super().__init__(Api.eval)

    async def register_benchmark(self, benchmark):
        return benchmark


class ToolGroupsImpl(Impl):
    def __init__(self):
        super().__init__(Api.tool_runtime)

    async def register_toolgroup(self, toolgroup: ToolGroup):
        return toolgroup

    async def unregister_toolgroup(self, toolgroup_id: str):
        return toolgroup_id

    async def list_runtime_tools(self, toolgroup_id, mcp_endpoint):
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="test-tool",
                    description="Test tool",
                    parameters=[ToolParameter(name="test-param", description="Test param", parameter_type="string")],
                )
            ]
        )


@pytest.mark.asyncio
async def test_models_routing_table(cached_disk_dist_registry):
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple models and verify listing
    await table.register_model(model_id="test-model", provider_id="test_provider")
    await table.register_model(model_id="test-model-2", provider_id="test_provider")

    models = await table.list_models()
    assert len(models.data) == 2
    model_ids = {m.identifier for m in models.data}
    assert "test-model" in model_ids
    assert "test-model-2" in model_ids

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 2
    openai_model_ids = {m.id for m in openai_models.data}
    assert "test-model" in openai_model_ids
    assert "test-model-2" in openai_model_ids

    # Test get_object_by_identifier
    model = await table.get_object_by_identifier("model", "test-model")
    assert model is not None
    assert model.identifier == "test-model"

    # Test get_object_by_identifier on non-existent object
    non_existent = await table.get_object_by_identifier("model", "non-existent-model")
    assert non_existent is None

    await table.unregister_model(model_id="test-model")
    await table.unregister_model(model_id="test-model-2")

    models = await table.list_models()
    assert len(models.data) == 0

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 0


@pytest.mark.asyncio
async def test_shields_routing_table(cached_disk_dist_registry):
    table = ShieldsRoutingTable({"test_provider": SafetyImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple shields and verify listing
    await table.register_shield(shield_id="test-shield", provider_id="test_provider")
    await table.register_shield(shield_id="test-shield-2", provider_id="test_provider")
    shields = await table.list_shields()

    assert len(shields.data) == 2
    shield_ids = {s.identifier for s in shields.data}
    assert "test-shield" in shield_ids
    assert "test-shield-2" in shield_ids


@pytest.mark.asyncio
async def test_vectordbs_routing_table(cached_disk_dist_registry):
    table = VectorDBsRoutingTable({"test_provider": VectorDBImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    m_table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await m_table.initialize()
    await m_table.register_model(
        model_id="test-model",
        provider_id="test_provider",
        metadata={"embedding_dimension": 128},
        model_type=ModelType.embedding,
    )

    # Register multiple vector databases and verify listing
    await table.register_vector_db(vector_db_id="test-vectordb", embedding_model="test-model")
    await table.register_vector_db(vector_db_id="test-vectordb-2", embedding_model="test-model")
    vector_dbs = await table.list_vector_dbs()

    assert len(vector_dbs.data) == 2
    vector_db_ids = {v.identifier for v in vector_dbs.data}
    assert "test-vectordb" in vector_db_ids
    assert "test-vectordb-2" in vector_db_ids

    await table.unregister_vector_db(vector_db_id="test-vectordb")
    await table.unregister_vector_db(vector_db_id="test-vectordb-2")

    vector_dbs = await table.list_vector_dbs()
    assert len(vector_dbs.data) == 0


async def test_datasets_routing_table(cached_disk_dist_registry):
    table = DatasetsRoutingTable({"localfs": DatasetsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple datasets and verify listing
    await table.register_dataset(
        dataset_id="test-dataset", purpose=DatasetPurpose.eval_messages_answer, source=URIDataSource(uri="test-uri")
    )
    await table.register_dataset(
        dataset_id="test-dataset-2", purpose=DatasetPurpose.eval_messages_answer, source=URIDataSource(uri="test-uri-2")
    )
    datasets = await table.list_datasets()

    assert len(datasets.data) == 2
    dataset_ids = {d.identifier for d in datasets.data}
    assert "test-dataset" in dataset_ids
    assert "test-dataset-2" in dataset_ids

    await table.unregister_dataset(dataset_id="test-dataset")
    await table.unregister_dataset(dataset_id="test-dataset-2")

    datasets = await table.list_datasets()
    assert len(datasets.data) == 0


@pytest.mark.asyncio
async def test_scoring_functions_routing_table(cached_disk_dist_registry):
    table = ScoringFunctionsRoutingTable({"test_provider": ScoringFunctionsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple scoring functions and verify listing
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn",
        provider_id="test_provider",
        description="Test scoring function",
        return_type=NumberType(),
    )
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn-2",
        provider_id="test_provider",
        description="Another test scoring function",
        return_type=NumberType(),
    )
    scoring_functions = await table.list_scoring_functions()

    assert len(scoring_functions.data) == 2
    scoring_fn_ids = {fn.identifier for fn in scoring_functions.data}
    assert "test-scoring-fn" in scoring_fn_ids
    assert "test-scoring-fn-2" in scoring_fn_ids


@pytest.mark.asyncio
async def test_benchmarks_routing_table(cached_disk_dist_registry):
    table = BenchmarksRoutingTable({"test_provider": BenchmarksImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple benchmarks and verify listing
    await table.register_benchmark(
        benchmark_id="test-benchmark",
        dataset_id="test-dataset",
        scoring_functions=["test-scoring-fn", "test-scoring-fn-2"],
    )
    benchmarks = await table.list_benchmarks()

    assert len(benchmarks.data) == 1
    benchmark_ids = {b.identifier for b in benchmarks.data}
    assert "test-benchmark" in benchmark_ids


@pytest.mark.asyncio
async def test_tool_groups_routing_table(cached_disk_dist_registry):
    table = ToolGroupsRoutingTable({"test_provider": ToolGroupsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple tool groups and verify listing
    await table.register_tool_group(
        toolgroup_id="test-toolgroup",
        provider_id="test_provider",
    )
    tool_groups = await table.list_tool_groups()

    assert len(tool_groups.data) == 1
    tool_group_ids = {tg.identifier for tg in tool_groups.data}
    assert "test-toolgroup" in tool_group_ids

    await table.unregister_toolgroup(toolgroup_id="test-toolgroup")
    tool_groups = await table.list_tool_groups()
    assert len(tool_groups.data) == 0


@pytest.mark.asyncio
async def test_tool_groups_routing_table_exception_handling(cached_disk_dist_registry):
    """Test that the tool group routing table handles exceptions when listing tools, like if an MCP server is unreachable."""

    exception_throwing_tool_groups_impl = ToolGroupsImpl()
    exception_throwing_tool_groups_impl.list_runtime_tools = AsyncMock(side_effect=Exception("Test exception"))

    table = ToolGroupsRoutingTable(
        {"test_provider": exception_throwing_tool_groups_impl}, cached_disk_dist_registry, {}
    )
    await table.initialize()

    await table.register_tool_group(
        toolgroup_id="test-toolgroup-exceptions",
        provider_id="test_provider",
        mcp_endpoint=URL(uri="http://localhost:8479/foo/bar"),
    )

    tools = await table.list_tools(toolgroup_id="test-toolgroup-exceptions")

    assert len(tools.data) == 0
