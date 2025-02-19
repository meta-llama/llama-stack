# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.operations import InsertOne, SearchIndexModel, UpdateOne
import certifi
from numpy.typing import NDArray

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO

from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)
  
from .config import MongoDBVectorIOConfig

from time import sleep

log = logging.getLogger(__name__)
CHUNK_ID_KEY = "_chunk_id"


class MongoDBAtlasIndex(EmbeddingIndex):

    def __init__(self, client: MongoClient, namespace: str, embeddings_key: str, embedding_dimension: str, index_name: str, filter_fields: List[str]):
        self.client = client
        self.namespace = namespace
        self.embeddings_key = embeddings_key
        self.index_name = index_name
        self.filter_fields = filter_fields
        self.embedding_dimension = embedding_dimension

    def _get_index_config(self, collection, index_name):
        idxs = list(collection.list_search_indexes())
        for ele in idxs:
            if ele["name"] == index_name:
                return ele

    def _get_search_index_model(self):
        index_fields = [
            {
                "path": self.embeddings_key,
                "type": "vector",
                        "numDimensions": self.embedding_dimension,
                        "similarity": "cosine"
            }
        ]

        if len(self.filter_fields) > 0:
            for filter_field in self.filter_fields:
                index_fields.append(
                    {
                        "path": filter_field,
                        "type": "filter"
                    }
                )

        return SearchIndexModel(
            name=self.index_name,
            type="vectorSearch",
            definition={
                "fields": index_fields
            }
        )

    def _check_n_create_index(self):
        client = self.client
        db, collection = self.namespace.split(".")
        collection = client[db][collection]
        index_name = self.index_name
        idx = self._get_index_config(collection, index_name)
        if not idx:
            log.info("Creating search index ...")
            search_index_model = self._get_search_index_model()
            collection.create_search_index(search_index_model)
            while True:
                idx = self._get_index_config(collection, index_name)
                if idx and idx["queryable"]:
                    log.info("Search index created successfully.")
                    break
                else:
                    log.info("Waiting for search index to be created ...")
                    sleep(5)
        else:
            log.info("Search index already exists.")

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        # Create a list of operations to perform in bulk
        operations = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{chunk.metadata['document_id']}:chunk-{i}"

            operations.append(
                UpdateOne(
                    {CHUNK_ID_KEY: chunk_id},
                    {
                        "$set": {
                            CHUNK_ID_KEY: chunk_id,
                            "chunk_content": json.loads(chunk.model_dump_json()),
                            self.embeddings_key: embedding.tolist(),
                        }
                    },
                    upsert=True,
                )
            )

        # Perform the bulk operations
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        collection.bulk_write(operations)
        print(f"Added {len(chunks)} chunks to the collection")
        # Create a search index model
        print("Creating search index ...")
        self._check_n_create_index()

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        # Perform a query
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]

        # Create vector search query
        vs_query = {"$vectorSearch":
                    {
                        "index": self.index_name,
                        "path": self.embeddings_key,
                        "queryVector": embedding.tolist(),
                        "numCandidates": k,
                        "limit": k,
                    }
                    }
        # Add a field to store the score
        score_add_field_query = {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"}
            }
        }
        if score_threshold is None:
            score_threshold = 0.01
        # Filter the results based on the score threshold
        filter_query = {
            "$match": {
                "score": {"$gt": score_threshold}
            }
        }

        project_query = {
            "$project": {
                CHUNK_ID_KEY: 1,
                "chunk_content": 1,
                "score": 1,
                "_id": 0,
            }
        }

        query = [vs_query, score_add_field_query, filter_query, project_query]

        results = collection.aggregate(query)

        # Create the response
        chunks = []
        scores = []
        for result in results:
            content = result["chunk_content"]
            chunk = Chunk(
                metadata=content["metadata"],
                content=content["content"],
            )
            chunks.append(chunk)
            scores.append(result["score"])

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        db, _ = self.namespace.split(".")
        self.client.drop_database(db)


class MongoDBVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: MongoDBVectorIOConfig, inference_api: Api.inference):
        self.config = config
        self.inference_api = inference_api
        self.cache = {}

    async def initialize(self) -> None:
        self.client = MongoClient(
            self.config.connection_str,
            tlsCAFile=certifi.where(),
        )

    async def shutdown(self) -> None:
        if not self.client:
            self.client.close()

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        index=MongoDBAtlasIndex(
                client=self.client,
                namespace=self.config.namespace,
                embeddings_key=self.config.embeddings_key,
                embedding_dimension=vector_db.embedding_dimension,
                index_name=self.config.index_name,
                filter_fields=self.config.filter_fields,
            )
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db=vector_db,
            index=index,
            inference_api=self.inference_api,
        )

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]
        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        self.cache[vector_db_id] = VectorDBWithIndex(
            vector_db=vector_db_id,
            index=MongoDBAtlasIndex(
                client=self.client,
                namespace=self.config.namespace,
                embeddings_key=self.config.embeddings_key,
                embedding_dimension=vector_db.embedding_dimension,
                index_name=self.config.index_name,
                filter_fields=self.config.filter_fields,
            ),
            inference_api=self.inference_api,
        )
        return self.cache[vector_db_id]

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(self,
                            vector_db_id: str,
                            chunks: List[Chunk],
                            ttl_seconds: Optional[int] = None,
                            ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        await index.insert_chunks(chunks)

    async def query_chunks(self,
                           vector_db_id: str,
                           query: InterleavedContent,
                           params: Optional[Dict[str, Any]] = None,
                           ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        return await index.query_chunks(query, params)
