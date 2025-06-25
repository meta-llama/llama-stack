# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.vector_io import (
    Chunk,
    ChunkMetadata,
    QueryChunksResponse,
)
from llama_stack.providers.inline.tool_runtime.rag.memory import MemoryToolRuntimeImpl


class TestRagQuery:
    @pytest.mark.asyncio
    async def test_query_raises_on_empty_vector_db_ids(self):
        rag_tool = MemoryToolRuntimeImpl(config=MagicMock(), vector_io_api=MagicMock(), inference_api=MagicMock())
        with pytest.raises(ValueError):
            await rag_tool.query(content=MagicMock(), vector_db_ids=[])

    @pytest.mark.asyncio
    async def test_query_chunk_metadata_handling(self):
        rag_tool = MemoryToolRuntimeImpl(config=MagicMock(), vector_io_api=MagicMock(), inference_api=MagicMock())
        content = "test query content"
        vector_db_ids = ["db1"]

        chunk_metadata = ChunkMetadata(
            document_id="doc1",
            chunk_id="chunk1",
            source="test_source",
            metadata_token_count=5,
        )
        interleaved_content = MagicMock()
        chunk = Chunk(
            content=interleaved_content,
            metadata={
                "key1": "value1",
                "token_count": 10,
                "metadata_token_count": 5,
                # Note this is inserted into `metadata` during MemoryToolRuntimeImpl().insert()
                "document_id": "doc1",
            },
            stored_chunk_id="chunk1",
            chunk_metadata=chunk_metadata,
        )

        query_response = QueryChunksResponse(chunks=[chunk], scores=[1.0])

        rag_tool.vector_io_api.query_chunks = AsyncMock(return_value=query_response)
        result = await rag_tool.query(content=content, vector_db_ids=vector_db_ids)

        assert result is not None
        expected_metadata_string = (
            "Metadata: {'key1': 'value1', 'document_id': 'doc1', 'chunk_id': 'chunk1', 'source': 'test_source'}"
        )
        assert expected_metadata_string in result.content[1].text
        assert result.content is not None
