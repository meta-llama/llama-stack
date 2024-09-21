# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api

from .routing_table import RoutingTable
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403

from types import MethodType

from termcolor import cprint


class MemoryRouter(Memory):
    """Routes to an provider based on the memory bank type"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.routing_table = routing_table

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        print("MemoryRouter: create_memory_bank")

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        print("MemoryRouter: get_memory_bank")

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        print("MemoryRouter: insert_documents")

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        print("query_documents")


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        self.api = Api.inference.value
        self.routing_table = routing_table

    async def initialize(self) -> None:
        await self.routing_table.initialize(self.api)

    async def shutdown(self) -> None:
        await self.routing_table.shutdown(self.api)

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = list,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        # TODO: we need to fix streaming response to align provider implementations with Protocol
        async for chunk in self.routing_table.get_provider_impl(
            self.api, model
        ).chat_completion(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
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
        return await self.routing_table.get_provider_impl(self.api, model).completion(
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
        return await self.routing_table.get_provider_impl(self.api, model).embeddings(
            model=model,
            contents=contents,
        )
