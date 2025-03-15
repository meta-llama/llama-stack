# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class VectorDB(Resource):
    type: Literal[ResourceType.vector_db.value] = ResourceType.vector_db.value

    embedding_model: str
    embedding_dimension: int

    @property
    def vector_db_id(self) -> str:
        return self.identifier

    @property
    def provider_vector_db_id(self) -> str:
        return self.provider_resource_id


class VectorDBInput(BaseModel):
    vector_db_id: str
    embedding_model: str
    embedding_dimension: int
    provider_vector_db_id: Optional[str] = None


class ListVectorDBsResponse(BaseModel):
    data: List[VectorDB]


@runtime_checkable
@trace_protocol
class VectorDBs(Protocol):
    @webmethod(route="/vector-dbs", method="GET")
    async def list_vector_dbs(self) -> ListVectorDBsResponse: ...

    @webmethod(route="/vector-dbs/{vector_db_id:path}", method="GET")
    async def get_vector_db(
        self,
        vector_db_id: str,
    ) -> VectorDB: ...

    @webmethod(route="/vector-dbs", method="POST")
    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: Optional[int] = 384,
        provider_id: Optional[str] = None,
        provider_vector_db_id: Optional[str] = None,
    ) -> VectorDB: ...

    @webmethod(route="/vector-dbs/{vector_db_id:path}", method="DELETE")
    async def unregister_vector_db(self, vector_db_id: str) -> None: ...
