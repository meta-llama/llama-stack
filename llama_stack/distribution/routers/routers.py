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
    """Routes to an provider based on the memory bank type"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table
        self.bank_id_to_type = {}

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def get_provider_from_bank_id(self, bank_id: str) -> Any:
        bank_type = self.bank_id_to_type.get(bank_id)
        if not bank_type:
            raise ValueError(f"Could not find bank type for {bank_id}")

        provider = self.routing_table.get_provider_impl(bank_type)
        if not provider:
            raise ValueError(f"Could not find provider for {bank_type}")
        return provider

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        bank_type = config.type
        bank = await self.routing_table.get_provider_impl(bank_type).create_memory_bank(
            name, config, url
        )
        self.bank_id_to_type[bank.bank_id] = bank_type
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        provider = self.get_provider_from_bank_id(bank_id)
        return await provider.get_memory_bank(bank_id)

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        return await self.get_provider_from_bank_id(bank_id).insert_documents(
            bank_id, documents, ttl_seconds
        )

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        return await self.get_provider_from_bank_id(bank_id).query_documents(
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

    async def chat_completion(
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
        # TODO: we need to fix streaming response to align provider implementations with Protocol.
        async for chunk in self.routing_table.get_provider_impl(model).chat_completion(
            **params
        ):
            yield chunk

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        return await self.routing_table.get_provider_impl(model).completion(
            model=model,
            content=content,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        )

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
