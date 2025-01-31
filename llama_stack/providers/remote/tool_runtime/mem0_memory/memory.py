# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging
import secrets
import string
import os
from typing import Any, Dict, List, Optional

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    TextContentItem,
    URL,
)
from llama_stack.apis.inference import Inference
from llama_stack.apis.tools import (
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    RAGToolRuntime,
    ToolDef,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.apis.vector_io import QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import ToolsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    content_from_doc,
    make_overlapped_chunks,
)

from .config import Mem0ToolRuntimeConfig
from llama_stack.providers.inline.tool_runtime.rag.context_retriever import generate_rag_query

import requests
from urllib.parse import urljoin
import json

log = logging.getLogger(__name__)


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


class Mem0MemoryToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime, RAGToolRuntime):
    def __init__(
        self,
        config: Mem0ToolRuntimeConfig,
        vector_io_api: VectorIO,
        inference_api: Inference,
    ):
        self.config = config
        self.vector_io_api = vector_io_api
        self.inference_api = inference_api

        # Mem0 API configuration
        self.api_base_url = config.host
        self.api_key = config.api_key or os.getenv("MEM0_API_KEY")
        self.org_id = config.org_id
        self.project_id = config.project_id

        # Validate configuration
        if not self.api_key:
            raise ValueError("Mem0 API Key not provided")
        if not (self.org_id and self.project_id):
            raise ValueError("Both org_id and project_id must be provided")

        # Setup headers
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        # Validate API key and connection
        self._validate_api_connection()

    def _validate_api_connection(self):
        """Validate API key and connection by making a test request."""
        try:
            params = {"org_id": self.org_id, "project_id": self.project_id}
            response = requests.get(
                urljoin(self.api_base_url, "/v1/ping/"),
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to validate Mem0 API connection: {str(e)}")

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def insert(
        self,
        documents: List[RAGDocument],
        vector_db_id: str,
        chunk_size_in_tokens: int = 512,
    ) -> None:
        chunks = []
        for doc in documents:
            content = await content_from_doc(doc)

            # Add to Mem0 memory via API
            try:
                payload = {
                    "messages": [{"role": "user", "content": content}],
                    "metadata": {"document_id": doc.document_id},
                    "org_id": self.org_id,
                    "project_id": self.project_id,
                    "user_id": vector_db_id,
                }

                response = requests.post(
                    urljoin(self.api_base_url, "/v1/memories/"),
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                log.error(f"Failed to insert document to Mem0: {str(e)}")
                # Continue with vector store insertion even if Mem0 fails

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

        # Search Mem0 memory via API
        mem0_chunks = []
        try:
            payload = {
                "query": query,
                "org_id": self.org_id,
                "project_id": self.project_id,
            }

            response = requests.post(
                urljoin(self.api_base_url, "/v1/memories/search/"),
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            mem0_results = response.json()
            mem0_chunks = [
                TextContentItem(
                    text=f"id:{result.get('metadata', {}).get('document_id', 'unknown')}; content:{result.get('memory', '')}"
                )
                for result in mem0_results
            ]
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to search Mem0: {str(e)}")
            # Continue with vector store search even if Mem0 fails

        if not mem0_chunks:
            return RAGQueryResult(content=None)

        return RAGQueryResult(
            content=[
                TextContentItem(
                    text="Here are the retrieved documents for relevant context:\n=== START-RETRIEVED-CONTEXT ===\n",
                ),
                *mem0_chunks,
                TextContentItem(
                    text="\n=== END-RETRIEVED-CONTEXT ===\n",
                ),
            ],
        )

    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        # Parameters are not listed since these methods are not yet invoked automatically
        # by the LLM. The method is only implemented so things like /tools can list without
        # encountering fatals.
        return [
            ToolDef(
                name="query_from_memory",
                description="Retrieve context from memory",
            ),
            ToolDef(
                name="insert_into_memory",
                description="Insert documents into memory",
            ),
        ]

    async def invoke_tool(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> ToolInvocationResult:
        raise RuntimeError(
            "This toolgroup should not be called generically but only through specific methods of the RAGToolRuntime protocol"
        )
