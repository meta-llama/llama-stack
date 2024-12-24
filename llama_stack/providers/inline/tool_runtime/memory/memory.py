# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import logging
import os
import re
import secrets
import string
import tempfile
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from llama_stack.apis.agents import Attachment
from llama_stack.apis.common.content_types import TextContentItem, URL
from llama_stack.apis.inference import Inference, InterleavedContent, Message
from llama_stack.apis.memory import Memory, MemoryBankDocument, QueryDocumentsResponse
from llama_stack.apis.memory_banks import MemoryBanks, VectorMemoryBankParams
from llama_stack.apis.tools import (
    ToolDef,
    ToolGroupDef,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.providers.datatypes import ToolsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.memory.vector_store import concat_interleaved_content
from pydantic import BaseModel

from .config import MemoryToolConfig
from .context_retriever import generate_rag_query

log = logging.getLogger(__name__)


class MemorySessionInfo(BaseModel):
    session_id: str
    session_name: str
    memory_bank_id: Optional[str] = None


def make_random_string(length: int = 8):
    return "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


class MemoryToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(
        self,
        config: MemoryToolConfig,
        memory_api: Memory,
        memory_banks_api: MemoryBanks,
        inference_api: Inference,
    ):
        self.config = config
        self.memory_api = memory_api
        self.memory_banks_api = memory_banks_api
        self.tempdir = tempfile.mkdtemp()
        self.inference_api = inference_api

    async def initialize(self):
        self.kvstore = await kvstore_impl(self.config.kvstore_config)

    async def discover_tools(self, tool_group: ToolGroupDef) -> List[ToolDef]:
        return []

    async def create_session(self, session_id: str) -> MemorySessionInfo:
        session_info = MemorySessionInfo(
            session_id=session_id,
            session_name=f"session_{session_id}",
        )
        await self.kvstore.set(
            key=f"memory::session:{session_id}",
            value=session_info.model_dump_json(),
        )
        return session_info

    async def get_session_info(self, session_id: str) -> Optional[MemorySessionInfo]:
        value = await self.kvstore.get(
            key=f"memory::session:{session_id}",
        )
        if not value:
            session_info = await self.create_session(session_id)
            return session_info

        return MemorySessionInfo(**json.loads(value))

    async def add_memory_bank_to_session(self, session_id: str, bank_id: str):
        session_info = await self.get_session_info(session_id)

        session_info.memory_bank_id = bank_id
        await self.kvstore.set(
            key=f"memory::session:{session_id}",
            value=session_info.model_dump_json(),
        )

    async def _ensure_memory_bank(self, session_id: str) -> str:
        session_info = await self.get_session_info(session_id)

        if session_info.memory_bank_id is None:
            bank_id = f"memory_bank_{session_id}"
            await self.memory_banks_api.register_memory_bank(
                memory_bank_id=bank_id,
                params=VectorMemoryBankParams(
                    embedding_model="all-MiniLM-L6-v2",
                    chunk_size_in_tokens=512,
                ),
            )
            await self.add_memory_bank_to_session(session_id, bank_id)
        else:
            bank_id = session_info.memory_bank_id

        return bank_id

    async def attachment_message(
        self, tempdir: str, urls: List[URL]
    ) -> List[TextContentItem]:
        content = []

        for url in urls:
            uri = url.uri
            if uri.startswith("file://"):
                filepath = uri[len("file://") :]
            elif uri.startswith("http"):
                path = urlparse(uri).path
                basename = os.path.basename(path)
                filepath = f"{tempdir}/{make_random_string() + basename}"
                log.info(f"Downloading {url} -> {filepath}")

                async with httpx.AsyncClient() as client:
                    r = await client.get(uri)
                    resp = r.text
                    with open(filepath, "w") as fp:
                        fp.write(resp)
            else:
                raise ValueError(f"Unsupported URL {url}")

            content.append(
                TextContentItem(
                    text=f'# There is a file accessible to you at "{filepath}"\n'
                )
            )

        return content

    async def _retrieve_context(
        self, session_id: str, messages: List[Message]
    ) -> Optional[List[InterleavedContent]]:
        bank_ids = []

        bank_ids.extend(c.bank_id for c in self.config.memory_bank_configs)

        session_info = await self.get_session_info(session_id)
        if session_info.memory_bank_id:
            bank_ids.append(session_info.memory_bank_id)

        if not bank_ids:
            # this can happen if the per-session memory bank is not yet populated
            # (i.e., no prior turns uploaded an Attachment)
            return None

        query = await generate_rag_query(
            self.config.query_generator_config,
            messages,
            inference_api=self.inference_api,
        )
        tasks = [
            self.memory_api.query_documents(
                bank_id=bank_id,
                query=query,
                params={
                    "max_chunks": 5,
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

    async def _process_attachments(
        self, session_id: str, attachments: List[Attachment]
    ):
        bank_id = await self._ensure_memory_bank(session_id)

        documents = [
            MemoryBankDocument(
                document_id=str(uuid.uuid4()),
                content=a.content,
                mime_type=a.mime_type,
                metadata={},
            )
            for a in attachments
            if isinstance(a.content, str)
        ]
        await self.memory_api.insert_documents(bank_id, documents)

        urls = [a.content for a in attachments if isinstance(a.content, URL)]
        # TODO: we need to migrate URL away from str type
        pattern = re.compile("^(https?://|file://|data:)")
        urls += [URL(uri=a.content) for a in attachments if pattern.match(a.content)]
        return await self.attachment_message(self.tempdir, urls)

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        if args["session_id"] is None:
            raise ValueError("session_id is required")

        context = await self._retrieve_context(
            args["session_id"], args["input_messages"]
        )
        if context is None:
            context = []
        attachments = args["attachments"]
        if attachments and len(attachments) > 0:
            context += await self._process_attachments(args["session_id"], attachments)
        return ToolInvocationResult(
            content=concat_interleaved_content(context), error_code=0
        )
