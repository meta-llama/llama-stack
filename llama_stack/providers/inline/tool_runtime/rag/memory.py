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

from pydantic import TypeAdapter

from llama_stack.apis.common.content_types import (
    URL,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import Inference
from llama_stack.apis.tools import (
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    Tool,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.apis.vector_io import QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import ToolsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    content_from_doc,
    make_overlapped_chunks,
)

from .config import RagToolRuntimeConfig
from .context_retriever import generate_rag_query

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


class MemoryToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime, RAGToolRuntime):
    def __init__(
        self,
        config: RagToolRuntimeConfig,
        vector_io_api: VectorIO,
        inference_api: Inference,
    ):
        self.config = config
        self.vector_io_api = vector_io_api
        self.inference_api = inference_api

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def register_tool(self, tool: Tool) -> None:
        pass

    async def unregister_tool(self, tool_id: str) -> None:
        return

    async def insert(
        self,
        documents: List[RAGDocument],
        vector_db_id: str,
        chunk_size_in_tokens: int = 512,
    ) -> None:
        chunks = []
        for doc in documents:
            content = await content_from_doc(doc)
            chunks.extend(
                make_overlapped_chunks(
                    doc.document_id,
                    content,
                    chunk_size_in_tokens,
                    chunk_size_in_tokens // 4,
                )
            )

        if not chunks:
            return

        await self.vector_io_api.insert_chunks(
            chunks=chunks,
            vector_db_id=vector_db_id,
        )

    async def query(
        self,
        content: InterleavedContent,
        vector_db_ids: List[str],
        query_config: Optional[RAGQueryConfig] = None,
    ) -> RAGQueryResult:
        if not vector_db_ids:
            return RAGQueryResult(content=None)

        query_config = query_config or RAGQueryConfig()
        query = await generate_rag_query(
            query_config.query_generator_config,
            content,
            inference_api=self.inference_api,
        )
        tasks = [
            self.vector_io_api.query_chunks(
                vector_db_id=vector_db_id,
                query=query,
                params={
                    "max_chunks": query_config.max_chunks,
                },
            )
            for vector_db_id in vector_db_ids
        ]
        results: List[QueryChunksResponse] = await asyncio.gather(*tasks)
        chunks = [c for r in results for c in r.chunks]
        scores = [s for r in results for s in r.scores]

        if not chunks:
            return RAGQueryResult(content=None)

        # sort by score
        chunks, scores = zip(*sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True), strict=False)  # type: ignore
        chunks = chunks[: query_config.max_chunks]

        tokens = 0
        picked: list[InterleavedContentItem] = [
            TextContentItem(
                text=f"knowledge_search tool found {len(chunks)} chunks:\nBEGIN of knowledge_search tool results.\n"
            )
        ]
        for i, c in enumerate(chunks):
            metadata = c.metadata
            tokens += metadata["token_count"]
            if tokens > query_config.max_tokens_in_context:
                log.error(
                    f"Using {len(picked)} chunks; reached max tokens in context: {tokens}",
                )
                break
            picked.append(
                TextContentItem(
                    text=f"Result {i + 1}:\nDocument_id:{metadata['document_id'][:5]}\nContent: {c.content}\n",
                )
            )
        picked.append(TextContentItem(text="END of knowledge_search tool results.\n"))

        return RAGQueryResult(
            content=picked,
            metadata={
                "document_ids": [c.metadata["document_id"] for c in chunks[: len(picked)]],
            },
        )

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        # Parameters are not listed since these methods are not yet invoked automatically
        # by the LLM. The method is only implemented so things like /tools can list without
        # encountering fatals.
        return [
            ToolDef(
                name="insert_into_memory",
                description="Insert documents into memory",
            ),
            ToolDef(
                name="knowledge_search",
                description="Search for information in a database.",
                parameters=[
                    ToolParameter(
                        name="query",
                        description="The query to search for. Can be a natural language sentence or keywords.",
                        parameter_type="string",
                    ),
                ],
            ),
        ]

    async def invoke_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> ToolInvocationResult:
        vector_db_ids = kwargs.get("vector_db_ids", [])
        query_config = kwargs.get("query_config")
        if query_config:
            query_config = TypeAdapter(RAGQueryConfig).validate_python(query_config)
        else:
            # handle someone passing an empty dict
            query_config = RAGQueryConfig()

        query = kwargs["query"]
        result = await self.query(
            content=query,
            vector_db_ids=vector_db_ids,
            query_config=query_config,
        )

        return ToolInvocationResult(
            content=result.content,
            metadata=result.metadata,
        )
