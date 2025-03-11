# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from pathlib import Path

import pytest

from llama_stack.apis.tools import RAGDocument
from llama_stack.providers.utils.memory.vector_store import URL, content_from_doc

DUMMY_PDF_PATH = Path(os.path.abspath(__file__)).parent / "fixtures" / "dummy.pdf"
# Depending on the machine, this can get parsed a couple of ways
DUMMY_PDF_TEXT_CHOICES = ["Dummy PDF file", "Dumm y PDF file"]


def read_file(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()


def data_url_from_file(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_returns_content_from_pdf_data_uri(self):
        data_uri = data_url_from_file(DUMMY_PDF_PATH)
        doc = RAGDocument(
            document_id="dummy",
            content=data_uri,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES

    @pytest.mark.asyncio
    async def test_downloads_pdf_and_returns_content(self):
        # Using GitHub to host the PDF file
        url = "https://raw.githubusercontent.com/meta-llama/llama-stack/da035d69cfca915318eaf485770a467ca3c2a238/llama_stack/providers/tests/memory/fixtures/dummy.pdf"
        doc = RAGDocument(
            document_id="dummy",
            content=url,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES

    @pytest.mark.asyncio
    async def test_downloads_pdf_and_returns_content_with_url_object(self):
        # Using GitHub to host the PDF file
        url = "https://raw.githubusercontent.com/meta-llama/llama-stack/da035d69cfca915318eaf485770a467ca3c2a238/llama_stack/providers/tests/memory/fixtures/dummy.pdf"
        doc = RAGDocument(
            document_id="dummy",
            content=URL(
                uri=url,
            ),
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content in DUMMY_PDF_TEXT_CHOICES
