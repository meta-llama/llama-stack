# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, Dict, List

from llama_stack.apis.datasetio.datasetio import DatasetIO
from llama_stack.distribution.datatypes import RoutingTable

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.apis.scoring import *  # noqa: F403


class MemoryRouter(Memory):
    """Routes to an provider based on the memory bank identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_memory_bank(self, memory_bank: MemoryBankDef) -> None:
        await self.routing_table.register_memory_bank(memory_bank)

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        return await self.routing_table.get_provider_impl(bank_id).insert_documents(
            bank_id, documents, ttl_seconds
        )

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        return await self.routing_table.get_provider_impl(bank_id).query_documents(
            bank_id, query, params
        )


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: ModelDef) -> None:
        await self.routing_table.register_model(model)

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        params = dict(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        provider = self.routing_table.get_provider_impl(model)
        if stream:
            return (chunk async for chunk in await provider.chat_completion(**params))
        else:
            return await provider.chat_completion(**params)

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        provider = self.routing_table.get_provider_impl(model)
        params = dict(
            model=model,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return (chunk async for chunk in await provider.completion(**params))
        else:
            return await provider.completion(**params)

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        return await self.routing_table.get_provider_impl(model).embeddings(
            model=model,
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

    async def register_shield(self, shield: ShieldDef) -> None:
        await self.routing_table.register_shield(shield)

    async def run_shield(
        self,
        shield_type: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        return await self.routing_table.get_provider_impl(shield_type).run_shield(
            shield_type=shield_type,
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
        return await self.routing_table.get_provider_impl(
            dataset_id
        ).get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=rows_in_page,
            page_token=page_token,
            filter_condition=filter_condition,
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
        scoring_functions: List[str],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse:
        res = {}
        for fn_identifier in scoring_functions:
            score_response = await self.routing_table.get_provider_impl(
                fn_identifier
            ).score_batch(
                dataset_id=dataset_id,
                scoring_functions=[fn_identifier],
            )
            res.update(score_response.results)

        if save_results_dataset:
            raise NotImplementedError("Save results dataset not implemented yet")

        return ScoreBatchResponse(
            results=res,
        )

    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse:
        res = {}
        # look up and map each scoring function to its provider impl
        for fn_identifier in scoring_functions:
            score_response = await self.routing_table.get_provider_impl(
                fn_identifier
            ).score(
                input_rows=input_rows,
                scoring_functions=[fn_identifier],
            )
            res.update(score_response.results)

        return ScoreResponse(results=res)
