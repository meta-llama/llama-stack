# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingInput,
    PreprocessingResponse,
    PreprocessingResult,
    Preprocessor,
    PreprocessorOptions,
)
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate
from llama_stack.providers.inline.preprocessing.docling import InlineDoclingConfig


class InclineDoclingPreprocessorImpl(Preprocessing, PreprocessorsProtocolPrivate):
    def __init__(self, config: InlineDoclingConfig) -> None:
        self.config = config
        self.converter = DocumentConverter()
        self.chunker = None

    async def initialize(self) -> None:
        if self.config.chunk:
            self.chunker = HybridChunker(tokenizer=self.config.tokenizer)

    async def shutdown(self) -> None: ...

    async def register_preprocessor(self, preprocessor: Preprocessor) -> None: ...

    async def unregister_preprocessor(self, preprocessor_id: str) -> None: ...

    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessingInput],
        options: PreprocessorOptions,
    ) -> PreprocessingResponse:
        results = []

        for inp in preprocessor_inputs:
            if isinstance(inp.path_or_content, str):
                url = inp.path_or_content
            elif isinstance(inp.path_or_content, URL):
                url = inp.path_or_content.uri
            else:
                raise ValueError(f"Unexpected type {type(inp.path_or_content)} for input {inp.path_or_content}")

            converted_document = self.converter.convert(url).document
            if self.config.chunk:
                result = self.chunker.chunk(converted_document)
                results.extend([PreprocessingResult(data=chunk.text, metadata=chunk.meta) for chunk in result])
            else:
                result = converted_document.export_to_markdown()
                results.append(result)

        return PreprocessingResponse(status=True, results=results)
