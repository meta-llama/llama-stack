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
from pytest_httpx import HTTPXMock

from llama_stack.apis.memory.memory import MemoryBankDocument, URL
from llama_stack.providers.utils.memory.vector_store import content_from_doc

DUMMY_PDF_PATH = Path(os.path.abspath(__file__)).parent / "fixtures" / "dummy.pdf"


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


# Requires pytest-httpx - pip install pytest-httpx
class TestVectorStore:
    @pytest.mark.asyncio
    async def test_returns_content_from_pdf_data_uri(self):
        data_uri = data_url_from_file(DUMMY_PDF_PATH)
        doc = MemoryBankDocument(
            document_id="dummy",
            content=data_uri,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content == "Dummy PDF file"

    @pytest.mark.asyncio
    async def test_downloads_pdf_and_returns_content(self, httpx_mock: HTTPXMock):
        url = "https://example.com/dummy.pdf"
        httpx_mock.add_response(url=url, content=read_file(DUMMY_PDF_PATH))
        doc = MemoryBankDocument(
            document_id="dummy",
            content=url,
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content == "Dummy PDF file"

    @pytest.mark.asyncio
    async def test_downloads_pdf_and_returns_content_with_url_object(
        self, httpx_mock: HTTPXMock
    ):
        url = "https://example.com/dummy.pdf"
        httpx_mock.add_response(url=url, content=read_file(DUMMY_PDF_PATH))
        doc = MemoryBankDocument(
            document_id="dummy",
            content=URL(
                uri=url,
            ),
            mime_type="application/pdf",
            metadata={},
        )
        content = await content_from_doc(doc)
        assert content == "Dummy PDF file"
