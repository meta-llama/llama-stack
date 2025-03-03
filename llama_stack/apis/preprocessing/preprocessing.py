# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.preprocessing.preprocessors import Preprocessor
from llama_stack.schema_utils import json_schema_type, webmethod


class PreprocessingInputType(Enum):
    document_content = "document_content"
    document_path = "document_path"


@json_schema_type
class PreprocessingInput(BaseModel):
    preprocessor_input_id: str
    preprocessor_input_type: Optional[PreprocessingInputType]
    path_or_content: str | URL


PreprocessorOptions = Dict[str, Any]

# TODO: shouldn't be just a string
PreprocessingResult = str


@json_schema_type
class PreprocessingResponse(BaseModel):
    status: bool
    results: Optional[List[str | PreprocessingResult]]


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
