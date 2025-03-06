# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.apis.preprocessing.preprocessors import Preprocessor
from llama_stack.apis.vector_io import Chunk
from llama_stack.schema_utils import json_schema_type, webmethod


class PreprocessingDataType(Enum):
    document_uri = "document_uri"
    document_directory_uri = "document_directory_uri"

    binary_document = "binary_document"
    raw_text_document = "raw_text_document"
    chunks = "chunks"


class PreprocessingDataFormat(Enum):
    pdf = "pdf"
    docx = "docx"
    xlsx = "xlsx"
    pptx = "pptx"
    md = "md"
    json = "json"
    html = "html"
    csv = "csv"


@json_schema_type
class PreprocessorInput(BaseModel):
    preprocessor_input_id: str
    preprocessor_input_type: Optional[PreprocessingDataType] = None
    preprocessor_input_format: Optional[PreprocessingDataFormat] = None
    path_or_content: str | InterleavedContent | URL


PreprocessorOptions = Dict[str, Any]


@json_schema_type
class PreprocessorChainElement(BaseModel):
    preprocessor_id: str
    options: Optional[PreprocessorOptions] = None


PreprocessorChain = List[PreprocessorChainElement]


@json_schema_type
class PreprocessorResponse(BaseModel):
    success: bool
    preprocessor_output_type: PreprocessingDataType
    results: Optional[List[str | InterleavedContent | Chunk]] = None


class PreprocessorStore(Protocol):
    def get_preprocessor(self, preprocessor_id: str) -> Preprocessor: ...


@runtime_checkable
class Preprocessing(Protocol):
    preprocessor_store: PreprocessorStore

    input_types: List[PreprocessingDataType]
    output_types: List[PreprocessingDataType]

    @webmethod(route="/preprocess", method="POST")
    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessorInput],
        options: Optional[PreprocessorOptions] = None,
    ) -> PreprocessorResponse: ...

    @webmethod(route="/chain_preprocess", method="POST")
    async def chain_preprocess(
        self,
        preprocessors: PreprocessorChain,
        preprocessor_inputs: List[PreprocessorInput],
    ) -> PreprocessorResponse: ...
