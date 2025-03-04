# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from enum import Enum
from typing import List, Tuple

from llama_models.llama3.api import Tokenizer

from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingDataType,
    PreprocessingInput,
    PreprocessingResponse,
    Preprocessor,
    PreprocessorOptions,
)
from llama_stack.apis.vector_io import Chunk
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate
from llama_stack.providers.inline.preprocessing.simple_chunking import InclineSimpleChunkingConfig

log = logging.getLogger(__name__)


class SimpleChunkingOptions(Enum):
    chunk_size_in_tokens = "chunk_size_in_tokens"
    chunk_overlap_ratio = "chunk_overlap_ratio"


class InclineSimpleChunkingImpl(Preprocessing, PreprocessorsProtocolPrivate):
    # this preprocessor receives plain text and returns chunks
    INPUT_TYPES = [PreprocessingDataType.raw_text_document]
    OUTPUT_TYPES = [PreprocessingDataType.chunks]

    def __init__(self, config: InclineSimpleChunkingConfig) -> None:
        self.config = config

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def register_preprocessor(self, preprocessor: Preprocessor) -> None: ...

    async def unregister_preprocessor(self, preprocessor_id: str) -> None: ...

    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessingInput],
        options: PreprocessorOptions,
    ) -> PreprocessingResponse:
        chunks = []

        window_len, overlap_len = self._resolve_chunk_size_params(options)

        for inp in preprocessor_inputs:
            new_chunks = self.make_overlapped_chunks(
                inp.preprocessor_input_id, inp.path_or_content, window_len, overlap_len
            )
            chunks.extend(new_chunks)

        return PreprocessingResponse(status=True, results=chunks)

    def _resolve_chunk_size_params(self, options: PreprocessorOptions) -> Tuple[int, int]:
        window_len = options.get(str(SimpleChunkingOptions.chunk_size_in_tokens), self.config.chunk_size_in_tokens)

        chunk_overlap_ratio = options.get(
            str(SimpleChunkingOptions.chunk_overlap_ratio), self.config.chunk_overlap_ratio
        )
        overlap_len = window_len // chunk_overlap_ratio

        return window_len, overlap_len

    @staticmethod
    def make_overlapped_chunks(document_id: str, text: str, window_len: int, overlap_len: int) -> List[Chunk]:
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
