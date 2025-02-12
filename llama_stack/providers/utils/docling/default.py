# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import io

from .chunker import Chunker
from .converter import Converter
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_stack.apis.vector_io import Chunk


class DefaultConverter(Converter):
    async def convert_from_data(self, data: bytes, encoding: str, mime_type: str | None) -> str:
        mime_category = mime_type.split("/")[0]
        if mime_category == "text":
            # For text-based files (including CSV, MD)
            return data.decode(encoding)

        elif mime_type == "application/pdf":
            return self.parse_pdf(data)

        else:
            log.error("Could not extract content from data_url properly.")
            return ""

    async def convert_from_url(self, data_url: str, mime_type: str | None) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get(data_url)
        if mime_type == "application/pdf":
            return self._parse_pdf(r.content)
        else:
            return r.text

    def _parse_pdf(self, data: bytes) -> str:
        from pypdf import PdfReader

        # For PDF and DOC/DOCX files, we can't reliably convert to string
        pdf_bytes = io.BytesIO(data)
        pdf_reader = PdfReader(pdf_bytes)
        return "\n".join([page.extract_text() for page in pdf_reader.pages])


class DefaultChunker(Chunker):
    def make_overlapped_chunks(self, document_id: str, text: str, window_len: int, overlap_len: int) -> list[Chunk]:
        tokenizer = Tokenizer.get_instance()
        tokens = tokenizer.encode(text, bos=False, eos=False)

        chunks = []
        for i in range(0, len(tokens), window_len - overlap_len):
            toks = tokens[i : i + window_len]
            chunk = tokenizer.decode(toks)
            # chunk is a string
            chunks.append(
                Chunk(
                    content=chunk,
                    metadata={
                        "token_count": len(toks),
                        "document_id": document_id,
                    },
                )
            )

        return chunks
