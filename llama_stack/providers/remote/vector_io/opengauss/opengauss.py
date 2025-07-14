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

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import OpenGaussVectorIOConfig

log = logging.getLogger(__name__)

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_dbs:opengauss:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:opengauss:{VERSION}::"


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
    def __init__(self, vector_db: VectorDB, dimension: int, conn, kvstore: KVStore | None = None):
        self.conn = conn
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            sanitized_identifier = vector_db.identifier.replace("-", "_")
            self.table_name = f"vector_store_{sanitized_identifier}"
            self.kvstore = kvstore

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
                    f"{chunk.chunk_id}",
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
                score = 1.0 / float(dist) if dist != 0 else float("inf")
                if score < score_threshold:
                    continue
                chunks.append(Chunk(**doc))
                scores.append(score)

            return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in OpenGauss")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in OpenGauss")

    async def delete(self):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove chunks from the OpenGauss table."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (chunk_ids,))


class OpenGaussVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: OpenGaussVectorIOConfig,
        inference_api: Any,
        files_api: Files | None = None,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.conn = None
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.files_api = files_api
        self.kvstore: KVStore | None = None
        self.vector_db_store = None
        self.openai_vector_store: dict[str, dict[str, Any]] = {}
        self.metadatadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        log.info(f"Initializing OpenGauss memory adapter with config: {self.config}")
        if self.config.kvstore is not None:
            self.kvstore = await kvstore_impl(self.config.kvstore)
        await self.initialize_openai_vector_stores()

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
                    log.info(f"OpenGauss server version: {version}")
                    log.info("Assuming native vector support is enabled in this OpenGauss instance.")

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS metadata_store (
                            key TEXT PRIMARY KEY,
                            data JSONB
                        )
                    """
                    )
        except Exception as e:
            log.exception("Could not connect to OpenGauss database server")
            raise RuntimeError("Could not connect to OpenGauss database server") from e

    async def shutdown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            log.info("Connection to OpenGauss database server closed")

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        assert self.kvstore is not None
        upsert_models(self.conn, [(vector_db.identifier, vector_db)])

        index = VectorDBWithIndex(
            vector_db,
            index=OpenGaussIndex(vector_db, vector_db.embedding_dimension, self.conn, kvstore=self.kvstore),
            inference_api=self.inference_api,
        )
        self.cache[vector_db.identifier] = index

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

        assert self.kvstore is not None
        await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_db_id}")

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
            raise RuntimeError("Vector DB store not initialized")

        vector_db = self.vector_db_store.get_vector_db(vector_db_id)
        if vector_db is None:
            raise VectorStoreNotFoundError(vector_db_id)

        index = OpenGaussIndex(vector_db, vector_db.embedding_dimension, self.conn)
        self.cache[vector_db_id] = VectorDBWithIndex(vector_db, index, self.inference_api)
        return self.cache[vector_db_id]

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from an OpenGauss vector store."""
        index = await self._get_and_cache_vector_db_index(store_id)
        if not index:
            raise VectorStoreNotFoundError(store_id)

        await index.index.delete_chunks(chunks_for_deletion)
