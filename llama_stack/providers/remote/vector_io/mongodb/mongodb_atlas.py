import pymongo

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

from pymongo import MongoClient,
from pymongo.operations import UpdateOne, InsertOne, DeleteOne, DeleteMany, SearchIndexModel
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

from .config import MongoDBAtlasVectorIOConfig
from time import sleep

log = logging.getLogger(__name__)
CHUNK_ID_KEY = "_chunk_id"

class MongoDBAtlasIndex(EmbeddingIndex):
    def __init__(self, client: MongoClient, namespace: str, embeddings_key: str, index_name: str):
        self.client = client
        self.namespace = namespace
        self.embeddings_key = embeddings_key
        self.index_name = index_name

    def _get_index_config(self, collection, index_name):
        idxs = list(collection.list_search_indexes())
        for ele in idxs:
            if ele["name"] == index_name:
                return ele

    def _check_n_create_index(self):
        client = self.client
        db,collection = self.namespace.split(".")
        collection = client[db][collection]
        index_name = self.index_name
        print(">>>>>>>>Index name: ", index_name, "<<<<<<<<<<")
        idx = self._get_index_config(collection, index_name)
        print(idx)
        if not idx:
            log.info("Creating search index ...")
            search_index_model = self._get_search_index_model()
            collection.create_search_index(search_index_model)
            while True:
                idx = self._get_index_config(collection, index_name)
                if idx and idx["queryable"]:
                    print("Search index created successfully.")
                    break
                else:
                    print("Waiting for search index to be created ...")
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
                InsertOne(
                    {
                        CHUNK_ID_KEY: chunk_id,
                        "chunk_content": chunk.model_dump_json(),
                        self.embeddings_key: embedding.tolist(),
                    }
                )
            )

        # Perform the bulk operations
        db,collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        collection.bulk_write(operations)

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        
        # Create a search index model
        self._check_n_create_index()

        # Perform a query
        db,collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]

        # Create vector search query
        vs_query = {"$vectorSearch": 
            {
                "index": "vector_index",
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
            chunk = Chunk(
                metadata={"document_id": result[CHUNK_ID_KEY]},
                content=json.loads(result["chunk_content"]),
            )
            chunks.append(chunk)
            scores.append(result["score"])

        return QueryChunksResponse(chunks=chunks, scores=scores)
    

class QdrantVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: MongoDBAtlasVectorIOConfig, inference_api: Api.inference):
        self.config = config
        self.inference_api = inference_api
    

    async def initialize(self) -> None:
        self.client = MongoClient(
            self.config.uri,
            tlsCAFile=certifi.where(),
        )
        self.cache = {}
        pass

    async def shutdown(self) -> None:
        self.client.close()
        pass

    async def register_vector_db( self, vector_db: VectorDB) -> None:
        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MongoDBAtlasIndex(
                client=self.client,
                namespace=self.config.namespace,
                embeddings_key=self.config.embeddings_key,
                index_name=self.config.index_name,
            ),
        )
        self.cache[vector_db] = index
        pass

    async def insert_chunks(self, vector_db_id, chunks, ttl_seconds = None):
        index = self.cache[vector_db_id].index
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        await index.insert_chunks(chunks)
    

    async def query_chunks(self, vector_db_id, query, params = None):
        index = self.cache[vector_db_id].index
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        return await index.query_chunks(query, params)
    
    


