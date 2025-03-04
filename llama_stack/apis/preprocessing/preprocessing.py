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


@json_schema_type
class PreprocessingInput(BaseModel):
    preprocessor_input_id: str
    preprocessor_input_type: Optional[PreprocessingDataType]
    path_or_content: str | URL


PreprocessorOptions = Dict[str, Any]


@json_schema_type
class PreprocessingResponse(BaseModel):
    status: bool
    results: Optional[List[str | InterleavedContent | Chunk]]


class PreprocessorStore(Protocol):
    def get_preprocessor(self, preprocessor_id: str) -> Preprocessor: ...


@runtime_checkable
class Preprocessing(Protocol):
    preprocessor_store: PreprocessorStore

    @webmethod(route="/preprocess", method="POST")
    async def preprocess(
        self,
        preprocessor_id: str,
        preprocessor_inputs: List[PreprocessingInput],
        options: PreprocessorOptions,
    ) -> PreprocessingResponse: ...
