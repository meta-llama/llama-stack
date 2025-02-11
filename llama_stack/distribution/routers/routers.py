# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import RawMessage
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.apis.datasetio import DatasetIO, PaginatedRowsResult
from llama_stack.apis.eval import (
    AppEvalTaskConfig,
    Eval,
    EvalTaskConfig,
    EvaluateResponse,
    Job,
    JobStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
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
from llama_stack.apis.telemetry import MetricEvent, Telemetry, TokenUsage
from llama_stack.apis.tools import (
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolDef,
    ToolRuntime,
)
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import RoutingTable
from llama_stack.providers.utils.telemetry.tracing import get_current_span


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: Optional[int] = 384,
        provider_id: Optional[str] = None,
        provider_vector_db_id: Optional[str] = None,
    ) -> None:
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
        return await self.routing_table.get_provider_impl(vector_db_id).insert_chunks(vector_db_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        return await self.routing_table.get_provider_impl(vector_db_id).query_chunks(vector_db_id, query, params)


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        self.routing_table = routing_table
        self.telemetry = telemetry
        if self.telemetry:
            self.tokenizer = Tokenizer.get_instance()
            self.formatter = ChatFormat(self.tokenizer)

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_model(
        self,
        model_id: str,
        provider_model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_type: Optional[ModelType] = None,
    ) -> None:
        await self.routing_table.register_model(model_id, provider_model_id, provider_id, metadata, model_type)

    async def _log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int, model: Model
    ) -> None:
        span = get_current_span()
        metrics = [
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        ]
        for metric_name, value in metrics:
            await self.telemetry.log_event(
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

    async def _add_token_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: Model,
        target: Any,
    ) -> None:
        metrics = getattr(target, "metrics", None)
        if metrics is None:
            target.metrics = []
        target.metrics.append(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        if self.telemetry:
            await self._log_token_usage(prompt_tokens, completion_tokens, total_tokens, model)

    async def _count_tokens(
        self,
        messages: Union[List[Message], List[RawMessage]],
        tool_prompt_format: Optional[ToolPromptFormat] = None,
    ) -> Optional[int]:
        if not self.telemetry:
            return None
        encoded = self.formatter.encode_dialog_prompt(messages, tool_prompt_format)
        return len(encoded.tokens) if encoded and encoded.tokens else 0

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        if model.model_type == ModelType.embedding:
            raise ValueError(f"Model '{model_id}' is an embedding model and does not support chat completions")
        if tool_config:
            if tool_choice != tool_config.tool_choice:
                raise ValueError("tool_choice and tool_config.tool_choice must match")
            if tool_prompt_format != tool_config.tool_prompt_format:
                raise ValueError("tool_prompt_format and tool_config.tool_prompt_format must match")
        else:
            tool_config = ToolConfig(
                tool_choice=tool_choice,
                tool_prompt_format=tool_prompt_format,
            )
        params = dict(
            model_id=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
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
                            [RawMessage(role="assistant", content=completion_text)],
                            tool_config.tool_prompt_format,
                        )
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                        await self._add_token_metrics(
                            prompt_tokens or 0,
                            completion_tokens or 0,
                            total_tokens,
                            model,
                            chunk,
                        )
                    yield chunk

            return stream_generator()
        else:
            response = await provider.chat_completion(**params)
            completion_tokens = await self._count_tokens(
                [response.completion_message],
                tool_config.tool_prompt_format,
            )
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            await self._add_token_metrics(
                prompt_tokens or 0,
                completion_tokens or 0,
                total_tokens,
                model,
                response,
            )
            return response

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
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

        prompt_tokens = await self._count_tokens([RawMessage(role="user", content=str(content))])

        if stream:

            async def stream_generator():
                completion_text = ""
                async for chunk in await provider.completion(**params):
                    if hasattr(chunk, "delta"):
                        completion_text += chunk.delta
                    if hasattr(chunk, "stop_reason") and chunk.stop_reason and self.telemetry:
                        completion_tokens = await self._count_tokens(
                            [RawMessage(role="assistant", content=completion_text)]
                        )
                        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                        await self._add_token_metrics(
                            prompt_tokens or 0,
                            completion_tokens or 0,
                            total_tokens,
                            model,
                            chunk,
                        )
                    yield chunk

            return stream_generator()
        else:
            response = await provider.completion(**params)
            completion_tokens = await self._count_tokens([RawMessage(role="assistant", content=str(response.content))])
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            await self._add_token_metrics(
                prompt_tokens or 0,
                completion_tokens or 0,
                total_tokens,
                model,
                response,
            )
            return response

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.routing_table.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        if model.model_type == ModelType.llm:
            raise ValueError(f"Model '{model_id}' is an LLM model and does not support embeddings")
        return await self.routing_table.get_provider_impl(model_id).embeddings(
            model_id=model_id,
            contents=contents,
        )


class SafetyRouter(Safety):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield:
        return await self.routing_table.register_shield(shield_id, provider_shield_id, provider_id, params)

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
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
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def get_rows_paginated(
        self,
        dataset_id: str,
        rows_in_page: int,
        page_token: Optional[str] = None,
        filter_condition: Optional[str] = None,
    ) -> PaginatedRowsResult:
        return await self.routing_table.get_provider_impl(dataset_id).get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=rows_in_page,
            page_token=page_token,
            filter_condition=filter_condition,
        )

    async def append_rows(self, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        return await self.routing_table.get_provider_impl(dataset_id).append_rows(
            dataset_id=dataset_id,
            rows=rows,
        )


class ScoringRouter(Scoring):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: Dict[str, Optional[ScoringFnParams]] = None,
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
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
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_eval(
        self,
        task_id: str,
        task_config: AppEvalTaskConfig,
    ) -> Job:
        return await self.routing_table.get_provider_impl(task_id).run_eval(
            task_id=task_id,
            task_config=task_config,
        )

    async def evaluate_rows(
        self,
        task_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: EvalTaskConfig,
    ) -> EvaluateResponse:
        return await self.routing_table.get_provider_impl(task_id).evaluate_rows(
            task_id=task_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            task_config=task_config,
        )

    async def job_status(
        self,
        task_id: str,
        job_id: str,
    ) -> Optional[JobStatus]:
        return await self.routing_table.get_provider_impl(task_id).job_status(task_id, job_id)

    async def job_cancel(
        self,
        task_id: str,
        job_id: str,
    ) -> None:
        await self.routing_table.get_provider_impl(task_id).job_cancel(
            task_id,
            job_id,
        )

    async def job_result(
        self,
        task_id: str,
        job_id: str,
    ) -> EvaluateResponse:
        return await self.routing_table.get_provider_impl(task_id).job_result(
            task_id,
            job_id,
        )


class ToolRuntimeRouter(ToolRuntime):
    class RagToolImpl(RAGToolRuntime):
        def __init__(
            self,
            routing_table: RoutingTable,
        ) -> None:
            self.routing_table = routing_table

        async def query(
            self,
            content: InterleavedContent,
            vector_db_ids: List[str],
            query_config: Optional[RAGQueryConfig] = None,
        ) -> RAGQueryResult:
            return await self.routing_table.get_provider_impl("query_from_memory").query(
                content, vector_db_ids, query_config
            )

        async def insert(
            self,
            documents: List[RAGDocument],
            vector_db_id: str,
            chunk_size_in_tokens: int = 512,
        ) -> None:
            return await self.routing_table.get_provider_impl("insert_into_memory").insert(
                documents, vector_db_id, chunk_size_in_tokens
            )

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

        # HACK ALERT this should be in sync with "get_all_api_endpoints()"
        self.rag_tool = self.RagToolImpl(routing_table)
        for method in ("query", "insert"):
            setattr(self, f"rag_tool.{method}", getattr(self.rag_tool, method))

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        return await self.routing_table.get_provider_impl(tool_name).invoke_tool(
            tool_name=tool_name,
            kwargs=kwargs,
        )

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        return await self.routing_table.get_provider_impl(tool_group_id).list_tools(tool_group_id, mcp_endpoint)
