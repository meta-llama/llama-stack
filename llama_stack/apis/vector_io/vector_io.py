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
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


class Chunk(BaseModel):
    content: InterleavedContent
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class QueryChunksResponse(BaseModel):
    chunks: List[Chunk]
    scores: List[float]


class VectorDBStore(Protocol):
    def get_vector_db(self, vector_db_id: str) -> Optional[VectorDB]: ...


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
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None: ...

    @webmethod(route="/vector-io/query", method="POST")
    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse: ...
