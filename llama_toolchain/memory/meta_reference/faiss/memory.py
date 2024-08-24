# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import faiss
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer


from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_toolchain.distribution.datatypes import Api, ProviderSpec
from llama_toolchain.memory.api import *  # noqa: F403
from .config import FaissImplConfig


async def get_provider_impl(config: FaissImplConfig, _deps: Dict[Api, ProviderSpec]):
    assert isinstance(
        config, FaissImplConfig
    ), f"Unexpected config type: {type(config)}"

    impl = FaissMemoryImpl(config)
    await impl.initialize()
    return impl


async def content_from_doc(doc: MemoryBankDocument) -> str:
    if isinstance(doc.content, URL):
        async with httpx.AsyncClient() as client:
            return await client.get(doc.content).text

    def _process(c):
        if isinstance(c, str):
            return c
        else:
            return "<media>"

    if isinstance(doc.content, list):
        return " ".join([_process(c) for c in doc.content])
    else:
        return _process(doc.content)


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


class BankState(BaseModel):
    bank: MemoryBank
    index: Optional[faiss.IndexFlatL2] = None
    doc_by_id: Dict[str, MemoryBankDocument] = Field(default_factory=dict)
    id_by_index: Dict[int, str] = Field(default_factory=dict)
    chunk_by_index: Dict[int, str] = Field(default_factory=dict)

    async def insert_documents(
        self,
        model: SentenceTransformer,
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
                )
                self.id_by_index[indexlen + i] = doc.document_id

    async def query_documents(
        self, model: SentenceTransformer, query: str, params: Dict[str, Any]
    ) -> Tuple[List[Chunk], List[float]]:
        k = params.get("max_chunks", 3)
        query_vector = model.encode([query])[0]
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )

        chunks = [self.chunk_by_index[int(i)] for i in indices[0]]
        scores = [1.0 / float(d) for d in distances[0]]

        return chunks, scores

    async def _ensure_index(self, dimension: int) -> faiss.IndexFlatL2:
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        return self.index


class FaissMemoryImpl(Memory):
    def __init__(self, config: FaissImplConfig) -> None:
        self.config = config
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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

        id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=id,
            name=name,
            config=config,
            url=url,
        )
        state = BankState(bank=bank)
        self.states[id] = state
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        if bank_id not in self.states:
            return None
        return self.states[bank_id].bank

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None:
        assert bank_id in self.states, f"Bank {bank_id} not found"
        state = self.states[bank_id]

        await state.insert_documents(self.model, documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        assert bank_id in self.states, f"Bank {bank_id} not found"
        state = self.states[bank_id]

        chunks, scores = await state.query_documents(self.model, query, params)
        return QueryDocumentsResponse(chunk=chunks, scores=scores)
