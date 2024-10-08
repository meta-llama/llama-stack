# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, Dict, List

from llama_stack.distribution.datatypes import RoutingTable

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403


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

    def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
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
            stream=stream,
            logprobs=logprobs,
        )
        provider = self.routing_table.get_provider_impl(model)
        if stream:
            return (chunk async for chunk in provider.chat_completion(**params))
        else:
            return provider.chat_completion(**params)

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        provider = self.routing_table.get_provider_impl(model)
        params = dict(
            model=model,
            content=content,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return (chunk async for chunk in provider.completion(**params))
        else:
            return provider.completion(**params)

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
