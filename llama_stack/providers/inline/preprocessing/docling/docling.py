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
    PreprocessingDataElement,
    PreprocessingDataFormat,
    PreprocessingDataType,
    Preprocessor,
    PreprocessorChain,
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
        self.converter = None
        self.chunker = None

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def register_preprocessor(self, preprocessor: Preprocessor) -> None: ...

    async def unregister_preprocessor(self, preprocessor_id: str) -> None: ...

    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessingDataElement],
        options: Optional[PreprocessorOptions] = None,
    ) -> PreprocessorResponse:
        if self.converter is None:
            # this is the first time this method is called
            self.converter = DocumentConverter()
            if self.config.chunk and self.chunker is None:
                # TODO: docling should use Llama Stack's inference API instead of handling tokenization by itself
                self.chunker = HybridChunker()

        results = []

        for inp in preprocessor_inputs:
            if isinstance(inp.data_element_path_or_content, str):
                url = inp.data_element_path_or_content
            elif isinstance(inp.data_element_path_or_content, URL):
                url = inp.data_element_path_or_content.uri
            else:
                log.error(
                    f"Unexpected type {type(inp.data_element_path_or_content)} for input {inp.data_element_path_or_content}, skipping this input."
                )
                continue

            converted_document = self.converter.convert(url).document

            if self.config.chunk:
                result = self.chunker.chunk(converted_document)
                for i, chunk in enumerate(result):
                    metadata = chunk.meta.dict()
                    # TODO: some vector DB adapters rely on a hard-coded header 'document_id'. This should be fixed.
                    metadata["document_id"] = inp.data_element_id
                    # TODO: the RAG tool implementation relies in a hard-coded header 'token_count'
                    metadata["token_count"] = self.chunker._count_chunk_tokens(chunk)
                    raw_chunk = Chunk(content=chunk.text, metadata=metadata)
                    chunk_data_element = PreprocessingDataElement(
                        data_element_id=f"{inp.data_element_id}_chunk_{i}",
                        data_element_type=PreprocessingDataType.chunks,
                        data_element_format=PreprocessingDataFormat.txt,
                        data_element_path_or_content=raw_chunk,
                    )
                    results.append(chunk_data_element)

            else:
                result = PreprocessingDataElement(
                    data_element_id=inp.data_element_id,
                    data_element_type=PreprocessingDataType.raw_text_document,
                    data_element_format=PreprocessingDataFormat.txt,
                    data_element_path_or_content=converted_document.export_to_markdown(),
                )
                results.append(result)

        output_data_type = (
            PreprocessingDataType.chunks if self.config.chunk else PreprocessingDataType.raw_text_document
        )
        return PreprocessorResponse(success=True, output_data_type=output_data_type, results=results)

    async def chain_preprocess(
        self,
        preprocessors: PreprocessorChain,
        preprocessor_inputs: List[PreprocessingDataElement],
    ) -> PreprocessorResponse:
        return await self.preprocess(preprocessor_id="", preprocessor_inputs=preprocessor_inputs)
