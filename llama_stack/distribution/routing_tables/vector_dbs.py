# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import TypeAdapter

from llama_stack.apis.models import ModelType
from llama_stack.apis.resource import ResourceType
from llama_stack.apis.vector_dbs import ListVectorDBsResponse, VectorDB, VectorDBs
from llama_stack.distribution.datatypes import (
    VectorDBWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core")


class VectorDBsRoutingTable(CommonRoutingTableImpl, VectorDBs):
    async def list_vector_dbs(self) -> ListVectorDBsResponse:
        return ListVectorDBsResponse(data=await self.get_all_with_type("vector_db"))

    async def get_vector_db(self, vector_db_id: str) -> VectorDB:
        vector_db = await self.get_object_by_identifier("vector_db", vector_db_id)
        if vector_db is None:
            raise ValueError(f"Vector DB '{vector_db_id}' not found")
        return vector_db

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorDB:
        if provider_vector_db_id is None:
            provider_vector_db_id = vector_db_id
        if provider_id is None:
            if len(self.impls_by_provider_id) > 0:
                provider_id = list(self.impls_by_provider_id.keys())[0]
                if len(self.impls_by_provider_id) > 1:
                    logger.warning(
                        f"No provider specified and multiple providers available. Arbitrarily selected the first provider {provider_id}."
                    )
            else:
                raise ValueError("No provider available. Please configure a vector_io provider.")
        model = await self.get_object_by_identifier("model", embedding_model)
        if model is None:
            raise ValueError(f"Model {embedding_model} not found")
        if model.model_type != ModelType.embedding:
            raise ValueError(f"Model {embedding_model} is not an embedding model")
        if "embedding_dimension" not in model.metadata:
            raise ValueError(f"Model {embedding_model} does not have an embedding dimension")
        vector_db_data = {
            "identifier": vector_db_id,
            "type": ResourceType.vector_db.value,
            "provider_id": provider_id,
            "provider_resource_id": provider_vector_db_id,
            "embedding_model": embedding_model,
            "embedding_dimension": model.metadata["embedding_dimension"],
        }
        vector_db = TypeAdapter(VectorDBWithOwner).validate_python(vector_db_data)
        await self.register_object(vector_db)
        return vector_db

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        existing_vector_db = await self.get_vector_db(vector_db_id)
        if existing_vector_db is None:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        await self.unregister_object(existing_vector_db)
