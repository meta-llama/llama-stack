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
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = list,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]:
        print("Inference Router: chat_completion")
