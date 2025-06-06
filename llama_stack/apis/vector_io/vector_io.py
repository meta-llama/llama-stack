# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


class Chunk(BaseModel):
    """
    A chunk of content that can be inserted into a vector database.
    :param content: The content of the chunk, which can be interleaved text, images, or other types.
    :param embedding: Optional embedding for the chunk. If not provided, it will be computed later.
    :param metadata: Metadata associated with the chunk, such as document ID, source, or other relevant information.
    """

    content: InterleavedContent
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


@json_schema_type
class QueryChunksResponse(BaseModel):
    chunks: list[Chunk]
    scores: list[float]


class VectorDBStore(Protocol):
    def get_vector_db(self, vector_db_id: str) -> VectorDB | None: ...


@runtime_checkable
@trace_protocol
class VectorIO(Protocol):
    vector_db_store: VectorDBStore | None = None

    # this will just block now until chunks are inserted, but it should
    # probably return a Job instance which can be polled for completion
    @webmethod(route="/vector-io/insert", method="POST")
    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Insert chunks into a vector database.

        :param vector_db_id: The identifier of the vector database to insert the chunks into.
        :param chunks: The chunks to insert. Each `Chunk` should contain content which can be interleaved text, images, or other types.
            `metadata`: `dict[str, Any]` and `embedding`: `List[float]` are optional.
            If `metadata` is provided, you configure how Llama Stack formats the chunk during generation.
            If `embedding` is not provided, it will be computed later.
        :param ttl_seconds: The time to live of the chunks.
        """
        ...

    @webmethod(route="/vector-io/query", method="POST")
    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from a vector database.

        :param vector_db_id: The identifier of the vector database to query.
        :param query: The query to search for.
        :param params: The parameters of the query.
        :returns: A QueryChunksResponse.
        """
        ...
