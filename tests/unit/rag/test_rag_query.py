# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock

import pytest

from llama_stack.apis.tools.rag_tool import RAGQueryConfig
from llama_stack.providers.inline.tool_runtime.rag.memory import MemoryToolRuntimeImpl


class TestRagQuery:
    @pytest.mark.asyncio
    async def test_query_raises_on_empty_vector_db_ids(self):
        rag_tool = MemoryToolRuntimeImpl(config=MagicMock(), vector_io_api=MagicMock(), inference_api=MagicMock())
        with pytest.raises(ValueError):
            await rag_tool.query(content=MagicMock(), vector_db_ids=[])

    @pytest.mark.asyncio
    async def test_query_raises_incorrect_mode(self):
        with pytest.raises(ValueError):
            RAGQueryConfig(mode="invalid_mode")

    @pytest.mark.asyncio
    async def test_query_accepts_valid_modes(self):
        RAGQueryConfig()  # Test default (vector)
        RAGQueryConfig(mode="vector")  # Test vector
        RAGQueryConfig(mode="keyword")  # Test keyword
        RAGQueryConfig(mode="hybrid")  # Test hybrid
