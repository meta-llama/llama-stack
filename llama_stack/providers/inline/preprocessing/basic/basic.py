# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
import re
from typing import List, Optional

import httpx

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingDataFormat,
    PreprocessingDataType,
    PreprocessingInput,
    PreprocessingResponse,
    Preprocessor,
    PreprocessorOptions,
)
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate
from llama_stack.providers.inline.preprocessing.basic.config import InlineBasicPreprocessorConfig
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.memory.vector_store import content_from_data, parse_pdf

log = logging.getLogger(__name__)


class InclineBasicPreprocessorImpl(Preprocessing, PreprocessorsProtocolPrivate):
    # this preprocessor can either receive documents (text or binary) or document URIs
    INPUT_TYPES = [
        PreprocessingDataType.binary_document,
        PreprocessingDataType.raw_text_document,
        PreprocessingDataType.document_uri,
    ]

    # this preprocessor optionally retrieves the documents and converts them into plain text
    OUTPUT_TYPES = [PreprocessingDataType.raw_text_document]

    URL_VALIDATION_PATTERN = re.compile("^(https?://|file://|data:)")

    def __init__(self, config: InlineBasicPreprocessorConfig) -> None:
        self.config = config

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def register_preprocessor(self, preprocessor: Preprocessor) -> None: ...

    async def unregister_preprocessor(self, preprocessor_id: str) -> None: ...

    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessingInput],
        options: Optional[PreprocessorOptions] = None,
    ) -> PreprocessingResponse:
        results = []

        for inp in preprocessor_inputs:
            input_type = self._resolve_input_type(inp)

            if input_type == PreprocessingDataType.document_uri:
                document = await self._fetch_document(inp)
                if document is None:
                    continue
            elif input_type == PreprocessingDataType.binary_document:
                document = inp.path_or_content
                if inp.preprocessor_input_format is None:
                    log.error(f"Binary document format is not provided for {inp.preprocessor_input_id}, skipping it")
                    continue
                if inp.preprocessor_input_format != PreprocessingDataFormat.pdf:
                    log.error(
                        f"Unsupported binary document type {inp.preprocessor_input_format} for {inp.preprocessor_input_id}, skipping it"
                    )
                    continue
            elif input_type == PreprocessingDataType.raw_text_document:
                document = interleaved_content_as_str(inp.path_or_content)
            else:
                log.error(f"Unexpected preprocessor input type: {inp.preprocessor_input_type}")
                continue

            if inp.preprocessor_input_format == PreprocessingDataFormat.pdf:
                document = parse_pdf(document)

            results.append(document)

        return PreprocessingResponse(status=True, results=results)

    @staticmethod
    async def _resolve_input_type(preprocessor_input: PreprocessingInput) -> PreprocessingDataType:
        if preprocessor_input.preprocessor_input_type is not None:
            return preprocessor_input.preprocessor_input_type

        if isinstance(preprocessor_input.path_or_content, URL):
            return PreprocessingDataType.document_uri
        if InclineBasicPreprocessorImpl.URL_VALIDATION_PATTERN.match(preprocessor_input.path_or_content):
            return PreprocessingDataType.document_uri
        if preprocessor_input.preprocessor_input_format == PreprocessingDataFormat.pdf:
            return PreprocessingDataType.binary_document

        return PreprocessingDataType.raw_text_document

    @staticmethod
    async def _fetch_document(preprocessor_input: PreprocessingInput) -> str | None:
        if isinstance(preprocessor_input.path_or_content, str):
            url = preprocessor_input.path_or_content
            if not InclineBasicPreprocessorImpl.URL_VALIDATION_PATTERN.match(url):
                log.error(f"Unexpected URL: {url}")
                return None
        elif isinstance(preprocessor_input.path_or_content, URL):
            url = preprocessor_input.path_or_content.uri
        else:
            log.error(
                f"Unexpected type {type(preprocessor_input.path_or_content)} for input {preprocessor_input.path_or_content}, skipping this input."
            )
            return None

        if url.startswith("data:"):
            return content_from_data(url)

        async with httpx.AsyncClient() as client:
            r = await client.get(url)

        return r.content if preprocessor_input.preprocessor_input_format == PreprocessingDataFormat.pdf else r.text

    @staticmethod
    def is_pdf(preprocessor_input: PreprocessingInput):
        return
