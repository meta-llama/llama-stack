# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Any

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import OpenGaussVectorIOConfig

log = logging.getLogger(__name__)


def upsert_models(conn, keys_models: list[tuple[str, BaseModel]]):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL(
            """
            MERGE INTO metadata_store AS target
            USING (VALUES %s) AS source (key, data)
            ON (target.key = source.key)
            WHEN MATCHED THEN
                UPDATE SET data = source.data
            WHEN NOT MATCHED THEN
                INSERT (key, data) VALUES (source.key, source.data);
            """
        )

        values = [(key, Json(model.model_dump())) for key, model in keys_models]
        execute_values(cur, query, values, template="(%s, %s::JSONB)")


def load_models(cur, cls):
    cur.execute("SELECT key, data FROM metadata_store")
    rows = cur.fetchall()
    return [TypeAdapter(cls).validate_python(row["data"]) for row in rows]


class OpenGaussIndex(EmbeddingIndex):
    def __init__(self, vector_db: VectorDB, dimension: int, conn):
        self.conn = conn
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            sanitized_identifier = vector_db.identifier.replace("-", "_")
            self.table_name = f"vector_store_{sanitized_identifier}"

            log.info(
                f"Creating table '{self.table_name}' for vector store '{vector_db.identifier}' if it does not exist."
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    document JSONB,
                    embedding vector({dimension})
                )
            """
            )

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        values = []
        for i, chunk in enumerate(chunks):
            values.append(
                (
                    f"{chunk.metadata['document_id']}:chunk-{i}",
                    Json(chunk.model_dump()),
                    embeddings[i].tolist(),
                )
            )

        query = sql.SQL(
            f"""
            MERGE INTO {self.table_name} AS target
            USING (VALUES %s) AS source (id, document, embedding)
            ON (target.id = source.id)
            WHEN MATCHED THEN
                UPDATE SET document = source.document, embedding = source.embedding
            WHEN NOT MATCHED THEN
                INSERT (id, document, embedding) VALUES (source.id, source.document, source.embedding);
            """
        )
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            execute_values(cur, query, values, template="(%s, %s::JSONB, %s::VECTOR)")

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
            SELECT document, embedding <=> %s::VECTOR AS distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT %s
        """,
                (embedding.tolist(), k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, dist in results:
                chunks.append(Chunk(**doc))
                scores.append(1.0 / float(dist) if dist != 0 else float("inf"))

            return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in this OpenGauss provider")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in this OpenGauss provider")

    async def delete(self):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")


class OpenGaussVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: OpenGaussVectorIOConfig, inference_api: Any) -> None:
        self.config = config
        self.inference_api = inference_api
        self.conn = None
        self.cache: dict[str, VectorDBWithIndex] = {}

    async def initialize(self) -> None:
        log.info(f"Initializing OpenGauss native vector adapter with config: {self.config}")
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
            )
            if self.conn:
                self.conn.autocommit = True
                with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    log.info(f"Successfully connected to OpenGauss. Server version: {version}")
                    log.info("Assuming native vector support is enabled in this OpenGauss instance.")

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS metadata_store (
                            key TEXT PRIMARY KEY,
                            data JSONB
                        )
                    """
                    )
            else:
                raise RuntimeError("Failed to establish a connection to the database.")
        except Exception as e:
            log.exception("Could not connect to OpenGauss database server")
            raise RuntimeError("Could not connect to OpenGauss database server") from e

    async def shutdown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            log.info("Connection to OpenGauss database server closed")

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        upsert_models(self.conn, [(vector_db.identifier, vector_db)])

        index = OpenGaussIndex(vector_db, vector_db.embedding_dimension, self.conn)
        self.cache[vector_db.identifier] = VectorDBWithIndex(vector_db, index, self.inference_api)

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        return await index.query_chunks(query, params)

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        if self.vector_db_store is None:
            raise RuntimeError("Vector DB store has not been initialized.")

        vector_db = self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB with id {vector_db_id} not found.")

        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        index = OpenGaussIndex(vector_db, vector_db.embedding_dimension, self.conn)
        self.cache[vector_db_id] = VectorDBWithIndex(vector_db, index, self.inference_api)
        return self.cache[vector_db_id]

    async def openai_create_vector_store(
        self,
        name: str,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorStoreObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = 10,
        ranking_options: SearchRankingOptions | None = None,
        rewrite_query: bool | None = False,
        search_mode: str | None = "vector",
    ) -> VectorStoreSearchResponsePage:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> VectorStoreFileObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentsResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> VectorStoreFileObject:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in OpenGauss")
