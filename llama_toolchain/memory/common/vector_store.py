# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from numpy.typing import NDArray

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_toolchain.memory.api import *  # noqa: F403


ALL_MINILM_L6_V2_DIMENSION = 384

EMBEDDING_MODEL = None


def get_embedding_model() -> "SentenceTransformer":
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        print("Loading sentence transformer")

        from sentence_transformers import SentenceTransformer

        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return EMBEDDING_MODEL


async def content_from_doc(doc: MemoryBankDocument) -> str:
    if isinstance(doc.content, URL):
        async with httpx.AsyncClient() as client:
            r = await client.get(doc.content.uri)
            return r.text

    return interleaved_text_media_as_str(doc.content)


def make_overlapped_chunks(
    document_id: str, text: str, window_len: int, overlap_len: int
) -> List[Chunk]:
    tokenizer = Tokenizer.get_instance()
    tokens = tokenizer.encode(text, bos=False, eos=False)

    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = tokenizer.decode(toks)
        chunks.append(
            Chunk(content=chunk, token_count=len(toks), document_id=document_id)
        )

    return chunks


class EmbeddingIndex(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        raise NotImplementedError()

    @abstractmethod
    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
        raise NotImplementedError()


@dataclass
class BankWithIndex:
    bank: MemoryBank
    index: EmbeddingIndex

    async def insert_documents(
        self,
        documents: List[MemoryBankDocument],
    ) -> None:
        model = get_embedding_model()
        for doc in documents:
            content = await content_from_doc(doc)
            chunks = make_overlapped_chunks(
                doc.document_id,
                content,
                self.bank.config.chunk_size_in_tokens,
                self.bank.config.overlap_size_in_tokens
                or (self.bank.config.chunk_size_in_tokens // 4),
            )
            embeddings = model.encode([x.content for x in chunks]).astype(np.float32)

            await self.index.add_chunks(chunks, embeddings)

    async def query_documents(
        self,
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

        model = get_embedding_model()
        query_vector = model.encode([query_str])[0].astype(np.float32)
        return await self.index.query(query_vector, k)
