# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from abc import ABC, abstractmethod

import chardet

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import RAGDocument
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)

from ...inline.tool_runtime.rag.config import DoclingConfig


class Converter(ABC):
    @abstractmethod
    async def convert_from_url(self, data_url: str, mime_type: str | None) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def convert_from_data(self, data: bytes, encoding: str, mime_type: str | None) -> str:
        raise NotImplementedError()

    @staticmethod
    def from_config(docling_config: DoclingConfig) -> "Converter":
        if docling_config:
            from .docling import DoclingConverter

            return DoclingConverter(docling_config)
        from .default import DefaultConverter

        return DefaultConverter()

    async def content_from_doc(self, doc: RAGDocument) -> str:
        doc_url: str | None = None
        if isinstance(doc.content, URL):
            doc_url = doc.content.uri
        else:
            pattern = re.compile("^(https?://|file://|data:)")
            if pattern.match(doc.content):
                doc_url = doc.content

        if doc_url:
            if doc_url.startswith("data:"):
                data, encoding, mime_type = self._decode_data(doc_url)
                # TODO probably not working with docling
                # Need to save the file first then use the converter
                return await self.convert_from_data(data, encoding, mime_type)
            else:
                return await self.convert_from_url(doc_url, doc.mime_type)

        return interleaved_content_as_str(doc.content)

    def _decode_data(self, data_url: str) -> (str, str, str):
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
        return data, encoding, mime_type
