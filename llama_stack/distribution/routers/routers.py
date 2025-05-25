# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
)
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse, Job
from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringFnParams,
)
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolRuntime,
)
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core")


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing VectorIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("VectorIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("VectorIORouter.shutdown")
        pass

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> None:
        logger.debug(f"VectorIORouter.register_vector_db: {vector_db_id}, {embedding_model}")
        await self.routing_table.register_vector_db(
            vector_db_id,
            embedding_model,
            embedding_dimension,
            provider_id,
            provider_vector_db_id,
        )

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        logger.debug(
            f"VectorIORouter.insert_chunks: {vector_db_id}, {len(chunks)} chunks, ttl_seconds={ttl_seconds}, chunk_ids={[chunk.metadata['document_id'] for chunk in chunks[:3]]}{' and more...' if len(chunks) > 3 else ''}",
        )
        return await self.routing_table.get_provider_impl(vector_db_id).insert_chunks(vector_db_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        logger.debug(f"VectorIORouter.query_chunks: {vector_db_id}")
        return await self.routing_table.get_provider_impl(vector_db_id).query_chunks(vector_db_id, query, params)


class ScoringRouter(Scoring):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing ScoringRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("ScoringRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("ScoringRouter.shutdown")
        pass

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: dict[str, ScoringFnParams | None] = None,
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        logger.debug(f"ScoringRouter.score_batch: {dataset_id}")
        res = {}
        for fn_identifier in scoring_functions.keys():
            score_response = await self.routing_table.get_provider_impl(fn_identifier).score_batch(
                dataset_id=dataset_id,
                scoring_functions={fn_identifier: scoring_functions[fn_identifier]},
            )
            res.update(score_response.results)

        if save_results_dataset:
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res,
        )

    async def score(
        self,
        input_rows: list[dict[str, Any]],
        scoring_functions: dict[str, ScoringFnParams | None] = None,
    ) -> ScoreResponse:
        logger.debug(f"ScoringRouter.score: {len(input_rows)} rows, {len(scoring_functions)} functions")
        res = {}
        # look up and map each scoring function to its provider impl
        for fn_identifier in scoring_functions.keys():
            score_response = await self.routing_table.get_provider_impl(fn_identifier).score(
                input_rows=input_rows,
                scoring_functions={fn_identifier: scoring_functions[fn_identifier]},
            )
            res.update(score_response.results)

        return ScoreResponse(results=res)


class EvalRouter(Eval):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing EvalRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("EvalRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("EvalRouter.shutdown")
        pass

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        logger.debug(f"EvalRouter.run_eval: {benchmark_id}")
        return await self.routing_table.get_provider_impl(benchmark_id).run_eval(
            benchmark_id=benchmark_id,
            benchmark_config=benchmark_config,
        )

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.evaluate_rows: {benchmark_id}, {len(input_rows)} rows")
        return await self.routing_table.get_provider_impl(benchmark_id).evaluate_rows(
            benchmark_id=benchmark_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )

    async def job_status(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> Job:
        logger.debug(f"EvalRouter.job_status: {benchmark_id}, {job_id}")
        return await self.routing_table.get_provider_impl(benchmark_id).job_status(benchmark_id, job_id)

    async def job_cancel(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> None:
        logger.debug(f"EvalRouter.job_cancel: {benchmark_id}, {job_id}")
        await self.routing_table.get_provider_impl(benchmark_id).job_cancel(
            benchmark_id,
            job_id,
        )

    async def job_result(
        self,
        benchmark_id: str,
        job_id: str,
    ) -> EvaluateResponse:
        logger.debug(f"EvalRouter.job_result: {benchmark_id}, {job_id}")
        return await self.routing_table.get_provider_impl(benchmark_id).job_result(
            benchmark_id,
            job_id,
        )


class ToolRuntimeRouter(ToolRuntime):
    class RagToolImpl(RAGToolRuntime):
        def __init__(
            self,
            routing_table: RoutingTable,
        ) -> None:
            logger.debug("Initializing ToolRuntimeRouter.RagToolImpl")
            self.routing_table = routing_table

        async def query(
            self,
            content: InterleavedContent,
            vector_db_ids: list[str],
            query_config: RAGQueryConfig | None = None,
        ) -> RAGQueryResult:
            logger.debug(f"ToolRuntimeRouter.RagToolImpl.query: {vector_db_ids}")
            return await self.routing_table.get_provider_impl("knowledge_search").query(
                content, vector_db_ids, query_config
            )

        async def insert(
            self,
            documents: list[RAGDocument],
            vector_db_id: str,
            chunk_size_in_tokens: int = 512,
        ) -> None:
            logger.debug(
                f"ToolRuntimeRouter.RagToolImpl.insert: {vector_db_id}, {len(documents)} documents, chunk_size={chunk_size_in_tokens}"
            )
            return await self.routing_table.get_provider_impl("insert_into_memory").insert(
                documents, vector_db_id, chunk_size_in_tokens
            )

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing ToolRuntimeRouter")
        self.routing_table = routing_table

        # HACK ALERT this should be in sync with "get_all_api_endpoints()"
        self.rag_tool = self.RagToolImpl(routing_table)
        for method in ("query", "insert"):
            setattr(self, f"rag_tool.{method}", getattr(self.rag_tool, method))

    async def initialize(self) -> None:
        logger.debug("ToolRuntimeRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("ToolRuntimeRouter.shutdown")
        pass

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        logger.debug(f"ToolRuntimeRouter.invoke_tool: {tool_name}")
        return await self.routing_table.get_provider_impl(tool_name).invoke_tool(
            tool_name=tool_name,
            kwargs=kwargs,
        )

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        logger.debug(f"ToolRuntimeRouter.list_runtime_tools: {tool_group_id}")
        return await self.routing_table.get_provider_impl(tool_group_id).list_tools(tool_group_id, mcp_endpoint)
