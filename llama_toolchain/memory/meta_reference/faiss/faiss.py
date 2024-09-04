# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import httpx
import numpy as np

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_toolchain.memory.api import *  # noqa: F403
from .config import FaissImplConfig


async def content_from_doc(doc: MemoryBankDocument) -> str:
    if isinstance(doc.content, URL):
        async with httpx.AsyncClient() as client:
            r = await client.get(doc.content.uri)
            return r.text

    return interleaved_text_media_as_str(doc.content)


def make_overlapped_chunks(
    text: str, window_len: int, overlap_len: int
) -> List[Tuple[str, int]]:
    tokenizer = Tokenizer.get_instance()
    tokens = tokenizer.encode(text, bos=False, eos=False)

    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = tokenizer.decode(toks)
        chunks.append((chunk, len(toks)))

    return chunks


@dataclass
class BankState:
    bank: MemoryBank
    index: Optional[faiss.IndexFlatL2] = None
    doc_by_id: Dict[str, MemoryBankDocument] = field(default_factory=dict)
    id_by_index: Dict[int, str] = field(default_factory=dict)
    chunk_by_index: Dict[int, str] = field(default_factory=dict)

    async def insert_documents(
        self,
        model: "SentenceTransformer",
        documents: List[MemoryBankDocument],
    ) -> None:
        tokenizer = Tokenizer.get_instance()
        chunk_size = self.bank.config.chunk_size_in_tokens

        for doc in documents:
            indexlen = len(self.id_by_index)
            self.doc_by_id[doc.document_id] = doc

            content = await content_from_doc(doc)
            chunks = make_overlapped_chunks(
                content,
                self.bank.config.chunk_size_in_tokens,
                self.bank.config.overlap_size_in_tokens
                or (self.bank.config.chunk_size_in_tokens // 4),
            )
            embeddings = model.encode([x[0] for x in chunks]).astype(np.float32)
            await self._ensure_index(embeddings.shape[1])

            self.index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self.chunk_by_index[indexlen + i] = Chunk(
                    content=chunk[0],
                    token_count=chunk[1],
                    document_id=doc.document_id,
                )
                print(f"Adding chunk #{indexlen + i} tokens={chunk[1]}")
                self.id_by_index[indexlen + i] = doc.document_id

    async def query_documents(
        self,
        model: "SentenceTransformer",
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        if params is None:
            params = {}
        k = params.get("max_chunks", 3)

        def _process(c) -> str:
            if isinstance(c, str):
                return c
            else:
                return "<media>"

        if isinstance(query, list):
            query_str = " ".join([_process(c) for c in query])
        else:
            query_str = _process(query)

        query_vector = model.encode([query_str])[0]
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )

        chunks = []
        scores = []
        for d, i in zip(distances[0], indices[0]):
            if i < 0:
                continue
            chunks.append(self.chunk_by_index[int(i)])
            scores.append(1.0 / float(d))

        return QueryDocumentsResponse(chunks=chunks, scores=scores)

    async def _ensure_index(self, dimension: int) -> faiss.IndexFlatL2:
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        return self.index


class FaissMemoryImpl(Memory):
    def __init__(self, config: FaissImplConfig) -> None:
        self.config = config
        self.model = None
        self.states = {}

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        assert url is None, "URL is not supported for this implementation"
        assert (
            config.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {config.type}"

        bank_id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=bank_id,
            name=name,
            config=config,
            url=url,
        )
        state = BankState(bank=bank)
        self.states[bank_id] = state
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        if bank_id not in self.states:
            return None
        return self.states[bank_id].bank

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        assert bank_id in self.states, f"Bank {bank_id} not found"
        state = self.states[bank_id]

        await state.insert_documents(self.get_model(), documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        assert bank_id in self.states, f"Bank {bank_id} not found"
        state = self.states[bank_id]

        return await state.query_documents(self.get_model(), query, params)

    def get_model(self) -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        if self.model is None:
            print("Loading sentence transformer")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        return self.model
