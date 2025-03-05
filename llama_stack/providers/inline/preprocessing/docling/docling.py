# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from typing import List, Optional

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingDataType,
    Preprocessor,
    PreprocessorChain,
    PreprocessorInput,
    PreprocessorOptions,
    PreprocessorResponse,
)
from llama_stack.apis.vector_io import Chunk
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate
from llama_stack.providers.inline.preprocessing.docling import InlineDoclingConfig

log = logging.getLogger(__name__)


class InclineDoclingPreprocessorImpl(Preprocessing, PreprocessorsProtocolPrivate):
    # this preprocessor receives URLs / paths to documents as input
    input_types = [PreprocessingDataType.document_uri]

    # this preprocessor either only converts the documents into a text format, or also chunks them
    output_types = [PreprocessingDataType.raw_text_document, PreprocessingDataType.chunks]

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
        preprocessor_inputs: List[PreprocessorInput],
        options: Optional[PreprocessorOptions] = None,
    ) -> PreprocessorResponse:
        results = []

        for inp in preprocessor_inputs:
            if isinstance(inp.path_or_content, str):
                url = inp.path_or_content
            elif isinstance(inp.path_or_content, URL):
                url = inp.path_or_content.uri
            else:
                log.error(
                    f"Unexpected type {type(inp.path_or_content)} for input {inp.path_or_content}, skipping this input."
                )
                continue

            converted_document = self.converter.convert(url).document
            if self.config.chunk:
                result = self.chunker.chunk(converted_document)
                results.extend([Chunk(content=chunk.text, metadata=chunk.meta) for chunk in result])
            else:
                result = converted_document.export_to_markdown()
                results.append(result)

        return PreprocessorResponse(status=True, results=results)

    async def chain_preprocess(
        self,
        preprocessors: PreprocessorChain,
        preprocessor_inputs: List[PreprocessorInput],
        is_rag_chain: Optional[bool] = False,
    ) -> PreprocessorResponse:
        return await self.preprocess(preprocessor_id="", preprocessor_inputs=preprocessor_inputs)
