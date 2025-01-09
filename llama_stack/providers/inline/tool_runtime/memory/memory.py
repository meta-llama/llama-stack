# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging
import secrets
import string
from typing import Any, Dict, List, Optional

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.memory import Memory, QueryDocumentsResponse
from llama_stack.apis.memory_banks import MemoryBanks
from llama_stack.apis.tools import (
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.providers.datatypes import ToolsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import concat_interleaved_content

from .config import MemoryToolConfig, MemoryToolRuntimeConfig
from .context_retriever import generate_rag_query

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


class MemoryToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(
        self,
        config: MemoryToolRuntimeConfig,
        memory_api: Memory,
        memory_banks_api: MemoryBanks,
        inference_api: Inference,
    ):
        self.config = config
        self.memory_api = memory_api
        self.memory_banks_api = memory_banks_api
        self.inference_api = inference_api

    async def initialize(self):
        pass

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        return [
            ToolDef(
                name="query_memory",
                description="Retrieve context from memory",
                parameters=[
                    ToolParameter(
                        name="messages",
                        description="The input messages to search for",
                        parameter_type="array",
                    ),
                ],
            )
        ]

    async def _retrieve_context(
        self, input_messages: List[InterleavedContent], bank_ids: List[str]
    ) -> Optional[List[InterleavedContent]]:
        if not bank_ids:
            return None
        query = await generate_rag_query(
            self.config.query_generator_config,
            input_messages,
            inference_api=self.inference_api,
        )
        tasks = [
            self.memory_api.query_documents(
                bank_id=bank_id,
                query=query,
                params={
                    "max_chunks": self.config.max_chunks,
                },
            )
            for bank_id in bank_ids
        ]
        results: List[QueryDocumentsResponse] = await asyncio.gather(*tasks)
        chunks = [c for r in results for c in r.chunks]
        scores = [s for r in results for s in r.scores]

        if not chunks:
            return None

        # sort by score
        chunks, scores = zip(
            *sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        )

        tokens = 0
        picked = []
        for c in chunks[: self.config.max_chunks]:
            tokens += c.token_count
            if tokens > self.config.max_tokens_in_context:
                log.error(
                    f"Using {len(picked)} chunks; reached max tokens in context: {tokens}",
                )
                break
            picked.append(f"id:{c.document_id}; content:{c.content}")

        return [
            "Here are the retrieved documents for relevant context:\n=== START-RETRIEVED-CONTEXT ===\n",
            *picked,
            "\n=== END-RETRIEVED-CONTEXT ===\n",
        ]

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        tool_group = await self.tool_store.get_tool_group(tool.toolgroup_id)
        final_args = tool_group.args or {}
        final_args.update(args)
        config = MemoryToolConfig()
        if tool.metadata and tool.metadata.get("config") is not None:
            config = MemoryToolConfig(**tool.metadata["config"])
        if "memory_bank_ids" in final_args:
            bank_ids = final_args["memory_bank_ids"]
        else:
            bank_ids = [
                bank_config.bank_id for bank_config in config.memory_bank_configs
            ]
        if "messages" not in final_args:
            raise ValueError("messages are required")
        context = await self._retrieve_context(
            final_args["messages"],
            bank_ids,
        )
        if context is None:
            context = []
        return ToolInvocationResult(
            content=concat_interleaved_content(context), error_code=0
        )
