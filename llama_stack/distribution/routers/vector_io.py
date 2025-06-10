# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import (
    InterleavedContent,
)
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core")


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing VectorIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("VectorIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("VectorIORouter.shutdown")
        pass

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> None:
        logger.debug(f"VectorIORouter.register_vector_db: {vector_db_id}, {embedding_model}")
        await self.routing_table.register_vector_db(
            vector_db_id,
            embedding_model,
            embedding_dimension,
            provider_id,
            provider_vector_db_id,
        )

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        logger.debug(
            f"VectorIORouter.insert_chunks: {vector_db_id}, {len(chunks)} chunks, ttl_seconds={ttl_seconds}, chunk_ids={[chunk.metadata['document_id'] for chunk in chunks[:3]]}{' and more...' if len(chunks) > 3 else ''}",
        )
        return await self.routing_table.get_provider_impl(vector_db_id).insert_chunks(vector_db_id, chunks, ttl_seconds)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        logger.debug(f"VectorIORouter.query_chunks: {vector_db_id}")
        return await self.routing_table.get_provider_impl(vector_db_id).query_chunks(vector_db_id, query, params)
