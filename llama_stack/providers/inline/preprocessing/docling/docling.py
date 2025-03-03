# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List

from llama_stack.apis.preprocessing import (
    Preprocessing,
    PreprocessingInput,
    PreprocessingResponse,
    Preprocessor,
    PreprocessorOptions,
)
from llama_stack.providers.datatypes import PreprocessorsProtocolPrivate
from llama_stack.providers.inline.preprocessing.docling import InlineDoclingConfig


class InclineDoclingPreprocessorImpl(Preprocessing, PreprocessorsProtocolPrivate):
    def __init__(self, config: InlineDoclingConfig) -> None:
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
    ) -> PreprocessingResponse: ...
