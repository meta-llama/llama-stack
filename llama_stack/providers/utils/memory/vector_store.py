# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import base64
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import chardet
import httpx
import numpy as np
from numpy.typing import NDArray
from pypdf import PdfReader

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.memory import *  # noqa: F403

ALL_MINILM_L6_V2_DIMENSION = 384

EMBEDDING_MODEL = None


def get_embedding_model() -> "SentenceTransformer":
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        print("Loading sentence transformer")

        from sentence_transformers import SentenceTransformer

        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return EMBEDDING_MODEL


def parse_data_url(data_url: str):
    data_url_pattern = re.compile(
        r"^"
        r"data:"
        r"(?P<mimetype>[\w/\-+.]+)"
        r"(?P<charset>;charset=(?P<encoding>[\w-]+))?"
        r"(?P<base64>;base64)?"
        r",(?P<data>.*)"
        r"$",
        re.DOTALL,
    )
    match = data_url_pattern.match(data_url)
    if not match:
        raise ValueError("Invalid Data URL format")

    parts = match.groupdict()
    parts["is_base64"] = bool(parts["base64"])
    return parts


def content_from_data(data_url: str) -> str:
    parts = parse_data_url(data_url)
    data = parts["data"]

    if parts["is_base64"]:
        data = base64.b64decode(data)
    else:
        data = unquote(data)
        encoding = parts["encoding"] or "utf-8"
        data = data.encode(encoding)

    encoding = parts["encoding"]
    if not encoding:
        detected = chardet.detect(data)
        encoding = detected["encoding"]

    mime_type = parts["mimetype"]
    mime_category = mime_type.split("/")[0]
    if mime_category == "text":
        # For text-based files (including CSV, MD)
        return data.decode(encoding)

    elif mime_type == "application/pdf":
        # For PDF and DOC/DOCX files, we can't reliably convert to string)
        pdf_bytes = io.BytesIO(data)
        pdf_reader = PdfReader(pdf_bytes)
        return "\n".join([page.extract_text() for page in pdf_reader.pages])

    else:
        cprint("Could not extract content from data_url properly.", color="red")
        return ""


async def content_from_doc(doc: MemoryBankDocument) -> str:
    if isinstance(doc.content, URL):
        if doc.content.uri.startswith("data:"):
            return content_from_data(doc.content.uri)
        else:
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
