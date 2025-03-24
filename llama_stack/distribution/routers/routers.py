# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.datasetio import DatasetIO, IterrowsResponse
from llama_stack.apis.datasets import DatasetPurpose, DataSource
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse, Job
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    StopReason,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.apis.safety import RunShieldResponse, Safety
from llama_stack.apis.scoring import (
    ScoreBatchResponse,
    ScoreResponse,
    Scoring,
    ScoringFnParams,
)
from llama_stack.apis.shields import Shield
from llama_stack.apis.telemetry import MetricEvent, MetricInResponse, Telemetry
from llama_stack.apis.tools import (
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolDef,
    ToolRuntime,
)
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.providers.datatypes import RoutingTable
from llama_stack.providers.utils.telemetry.tracing import get_current_span

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
        embedding_dimension: Optional[int] = 384,
        provider_id: Optional[str] = None,
        provider_vector_db_id: Optional[str] = None,
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
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        logger.debug(
            f"VectorIORouter.insert_chunks: {vector_db_id}, {len(chunks)} chunks, ttl_seconds={ttl_seconds}, chunk_ids={[chunk.metadata['document_id'] for chunk in chunks[:3]]}{' and more...' if len(chunks) > 3 else ''}",
        )
        return await self.routing_table.get_provider_impl(vector_db_id).insert_chunks(vector_db_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        logger.debug(f"VectorIORouter.query_chunks: {vector_db_id}")
        return await self.routing_table.get_provider_impl(vector_db_id).query_chunks(vector_db_id, query, params)


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        logger.debug("Initializing InferenceRouter")
        self.routing_table = routing_table
        self.telemetry = telemetry
        if self.telemetry:
            self.tokenizer = Tokenizer.get_instance()
            self.formatter = ChatFormat(self.tokenizer)

    async def initialize(self) -> None:
        logger.debug("InferenceRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("InferenceRouter.shutdown")
        pass

    async def register_model(
        self,
        model_id: str,
        provider_model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_type: Optional[ModelType] = None,
    ) -> None:
        logger.debug(
            f"InferenceRouter.register_model: {model_id=} {provider_model_id=} {provider_id=} {metadata=} {model_type=}",
        )
        await self.routing_table.register_model(model_id, provider_model_id, provider_id, metadata, model_type)

    def _construct_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: Model,
    ) -> List[MetricEvent]:
        """Constructs a list of MetricEvent objects containing token usage metrics.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens used
            model: Model object containing model_id and provider_id

        Returns:
            List of MetricEvent objects with token usage metrics
        """
        span = get_current_span()
        if span is None:
            logger.warning("No span found for token usage metrics")
            return []
        metrics = [
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ]
        metric_events = []
        for metric_name, value in metrics:
            metric_events.append(
                MetricEvent(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    metric=metric_name,
                    value=value,
                    timestamp=time.time(),
                    unit="tokens",
                    attributes={
                        "model_id": model.model_id,
                        "provider_id": model.provider_id,
                    },
                )
            )
        return metric_events

    async def _compute_and_log_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: Model,
    ) -> List[MetricInResponse]:
        metrics = self._construct_metrics(prompt_tokens, completion_tokens, total_tokens, model)
        if self.telemetry:
            for metric in metrics:
                await self.telemetry.log_event(metric)
        return [MetricInResponse(metric=metric.metric, value=metric.value) for metric in metrics]

    async def _count_tokens(
        self,
        messages: List[Message] | InterleavedContent,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
    ) -> Optional[int]:
        if isinstance(messages, list):
            encoded = self.formatter.encode_dialog_prompt(messages, tool_prompt_format)
        else:
            encoded = self.formatter.encode_content(messages)
        return len(encoded.tokens) if encoded and encoded.tokens else 0

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        logger.debug(
            f"InferenceRouter.chat_completion: {model_id=}, {stream=}, {messages=}, {tools=}, {tool_config=}, {response_format=}",
        )
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        if model.model_type == ModelType.embedding:
            raise ValueError(f"Model '{model_id}' is an embedding model and does not support chat completions")
        if tool_config:
            if tool_choice and tool_choice != tool_config.tool_choice:
                raise ValueError("tool_choice and tool_config.tool_choice must match")
            if tool_prompt_format and tool_prompt_format != tool_config.tool_prompt_format:
                raise ValueError("tool_prompt_format and tool_config.tool_prompt_format must match")
        else:
            params = {}
            if tool_choice:
                params["tool_choice"] = tool_choice
            if tool_prompt_format:
                params["tool_prompt_format"] = tool_prompt_format
            tool_config = ToolConfig(**params)

        tools = tools or []
        if tool_config.tool_choice == ToolChoice.none:
            tools = []
        elif tool_config.tool_choice == ToolChoice.auto:
            pass
        elif tool_config.tool_choice == ToolChoice.required:
            pass
        else:
            # verify tool_choice is one of the tools
            tool_names = [t.tool_name if isinstance(t.tool_name, str) else t.tool_name.value for t in tools]
            if tool_config.tool_choice not in tool_names:
                raise ValueError(f"Tool choice {tool_config.tool_choice} is not one of the tools: {tool_names}")

        params = dict(
            model_id=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )
        provider = self.routing_table.get_provider_impl(model_id)
        prompt_tokens = await self._count_tokens(messages, tool_config.tool_prompt_format)

        if stream:

            async def stream_generator():
                completion_text = ""
                async for chunk in await provider.chat_completion(**params):
                    if chunk.event.event_type == ChatCompletionResponseEventType.progress:
                        if chunk.event.delta.type == "text":
                            completion_text += chunk.event.delta.text
                    if chunk.event.event_type == ChatCompletionResponseEventType.complete:
                        completion_tokens = await self._count_tokens(
                            [
                                CompletionMessage(
                                    content=completion_text,
                                    stop_reason=StopReason.end_of_turn,
                                )
                            ],
                            tool_config.tool_prompt_format,
                        )
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                        metrics = await self._compute_and_log_token_usage(
                            prompt_tokens or 0,
                            completion_tokens or 0,
                            total_tokens,
                            model,
                        )
                        chunk.metrics = metrics if chunk.metrics is None else chunk.metrics + metrics
                    yield chunk

            return stream_generator()
        else:
            response = await provider.chat_completion(**params)
            completion_tokens = await self._count_tokens(
                [response.completion_message],
                tool_config.tool_prompt_format,
            )
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            metrics = await self._compute_and_log_token_usage(
                prompt_tokens or 0,
                completion_tokens or 0,
                total_tokens,
                model,
            )
            response.metrics = metrics if response.metrics is None else response.metrics + metrics
            return response

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        logger.debug(
            f"InferenceRouter.completion: {model_id=}, {stream=}, {content=}, {sampling_params=}, {response_format=}",
        )
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        if model.model_type == ModelType.embedding:
            raise ValueError(f"Model '{model_id}' is an embedding model and does not support chat completions")
        provider = self.routing_table.get_provider_impl(model_id)
        params = dict(
            model_id=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        prompt_tokens = await self._count_tokens(content)

        if stream:

            async def stream_generator():
                completion_text = ""
                async for chunk in await provider.completion(**params):
                    if hasattr(chunk, "delta"):
                        completion_text += chunk.delta
                    if hasattr(chunk, "stop_reason") and chunk.stop_reason and self.telemetry:
                        completion_tokens = await self._count_tokens(completion_text)
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                        metrics = await self._compute_and_log_token_usage(
                            prompt_tokens or 0,
                            completion_tokens or 0,
                            total_tokens,
                            model,
                        )
                        chunk.metrics = metrics if chunk.metrics is None else chunk.metrics + metrics
                    yield chunk

            return stream_generator()
        else:
            response = await provider.completion(**params)
            completion_tokens = await self._count_tokens(response.content)
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            metrics = await self._compute_and_log_token_usage(
                prompt_tokens or 0,
                completion_tokens or 0,
                total_tokens,
                model,
            )
            response.metrics = metrics if response.metrics is None else response.metrics + metrics
            return response

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        logger.debug(f"InferenceRouter.embeddings: {model_id}")
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        if model.model_type == ModelType.llm:
            raise ValueError(f"Model '{model_id}' is an LLM model and does not support embeddings")
        return await self.routing_table.get_provider_impl(model_id).embeddings(
            model_id=model_id,
            contents=contents,
            text_truncation=text_truncation,
            output_dimension=output_dimension,
            task_type=task_type,
        )


class SafetyRouter(Safety):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing SafetyRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("SafetyRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("SafetyRouter.shutdown")
        pass

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield:
        logger.debug(f"SafetyRouter.register_shield: {shield_id}")
        return await self.routing_table.register_shield(shield_id, provider_shield_id, provider_id, params)

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        logger.debug(f"SafetyRouter.run_shield: {shield_id}")
        return await self.routing_table.get_provider_impl(shield_id).run_shield(
            shield_id=shield_id,
            messages=messages,
            params=params,
        )


class DatasetIORouter(DatasetIO):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing DatasetIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("DatasetIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("DatasetIORouter.shutdown")
        pass

    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> None:
        logger.debug(
            f"DatasetIORouter.register_dataset: {purpose=} {source=} {metadata=} {dataset_id=}",
        )
        await self.routing_table.register_dataset(
            purpose=purpose,
            source=source,
            metadata=metadata,
            dataset_id=dataset_id,
        )

    async def iterrows(
        self,
        dataset_id: str,
        start_index: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> IterrowsResponse:
        logger.debug(
            f"DatasetIORouter.iterrows: {dataset_id}, {start_index=} {limit=}",
        )
        return await self.routing_table.get_provider_impl(dataset_id).iterrows(
            dataset_id=dataset_id,
            start_index=start_index,
            limit=limit,
        )

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        logger.debug(f"DatasetIORouter.append_rows: {dataset_id}, {len(rows)} rows")
        return await self.routing_table.get_provider_impl(dataset_id).append_rows(
            dataset_id=dataset_id,
            rows=rows,
        )


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
        scoring_functions: Dict[str, Optional[ScoringFnParams]] = None,
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
        input_rows: List[Dict[str, Any]],
        scoring_functions: Dict[str, Optional[ScoringFnParams]] = None,
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
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
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
            vector_db_ids: List[str],
            query_config: Optional[RAGQueryConfig] = None,
        ) -> RAGQueryResult:
            logger.debug(f"ToolRuntimeRouter.RagToolImpl.query: {vector_db_ids}")
            return await self.routing_table.get_provider_impl("knowledge_search").query(
                content, vector_db_ids, query_config
            )

        async def insert(
            self,
            documents: List[RAGDocument],
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

    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        logger.debug(f"ToolRuntimeRouter.invoke_tool: {tool_name}")
        return await self.routing_table.get_provider_impl(tool_name).invoke_tool(
            tool_name=tool_name,
            kwargs=kwargs,
        )

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        logger.debug(f"ToolRuntimeRouter.list_runtime_tools: {tool_group_id}")
        return await self.routing_table.get_provider_impl(tool_group_id).list_tools(tool_group_id, mcp_endpoint)
