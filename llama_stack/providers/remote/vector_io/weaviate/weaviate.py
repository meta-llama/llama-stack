# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from numpy.typing import NDArray
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import WeaviateRequestProviderData, WeaviateVectorIOConfig

log = logging.getLogger(__name__)


class WeaviateIndex(EmbeddingIndex):
    def __init__(self, client: weaviate.Client, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        data_objects = []
        for i, chunk in enumerate(chunks):
            data_objects.append(
                wvc.data.DataObject(
                    properties={
                        "chunk_content": chunk.model_dump_json(),
                    },
                    vector=embeddings[i].tolist(),
                )
            )

        # Inserting chunks into a prespecified Weaviate collection
        collection = self.client.collections.get(self.collection_name)

        # TODO: make this async friendly
        collection.data.insert_many(data_objects)

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        collection = self.client.collections.get(self.collection_name)

        results = collection.query.near_vector(
            near_vector=embedding.tolist(),
            limit=k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

        chunks = []
        scores = []
        for doc in results.objects:
            chunk_json = doc.properties["chunk_content"]
            try:
                chunk_dict = json.loads(chunk_json)
                chunk = Chunk(**chunk_dict)
            except Exception:
                log.exception(f"Failed to parse document: {chunk_json}")
                continue

            chunks.append(chunk)
            scores.append(1.0 / doc.metadata.distance)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self, chunk_ids: List[str]) -> None:
        collection = self.client.collections.get(self.collection_name)
        collection.data.delete_many(where=Filter.by_property("id").contains_any(chunk_ids))


class WeaviateVectorIOAdapter(
    VectorIO,
    NeedsRequestProviderData,
    VectorDBsProtocolPrivate,
):
    def __init__(self, config: WeaviateVectorIOConfig, inference_api: Api.inference) -> None:
        self.config = config
        self.inference_api = inference_api
        self.client_cache = {}
        self.cache = {}

    def _get_client(self) -> weaviate.Client:
        provider_data = self.get_request_provider_data()
        assert provider_data is not None, "Request provider data must be set"
        assert isinstance(provider_data, WeaviateRequestProviderData)

        key = f"{provider_data.weaviate_cluster_url}::{provider_data.weaviate_api_key}"
        if key in self.client_cache:
            return self.client_cache[key]

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=provider_data.weaviate_cluster_url,
            auth_credentials=Auth.api_key(provider_data.weaviate_api_key),
        )
        self.client_cache[key] = client
        return client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        for client in self.client_cache.values():
            client.close()

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        client = self._get_client()

        # Create collection if it doesn't exist
        if not client.collections.exists(vector_db.identifier):
            client.collections.create(
                name=vector_db.identifier,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(
                        name="chunk_content",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                ],
            )

        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db,
            WeaviateIndex(client=client, collection_name=vector_db.identifier),
            self.inference_api,
        )

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> Optional[VectorDBWithIndex]:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        client = self._get_client()
        if not client.collections.exists(vector_db.identifier):
            raise ValueError(f"Collection with name `{vector_db.identifier}` not found")

        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=WeaviateIndex(client=client, collection_name=vector_db.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_db_id] = index
        return index

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        return await index.query_chunks(query, params)
