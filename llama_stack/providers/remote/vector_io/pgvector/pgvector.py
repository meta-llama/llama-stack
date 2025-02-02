# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import execute_values, Json

from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate

from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import PGVectorConfig

log = logging.getLogger(__name__)


def check_extension_version(cur):
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    return result[0] if result else None


def upsert_models(cur, keys_models: List[Tuple[str, BaseModel]]):
    query = sql.SQL(
        """
        INSERT INTO metadata_store (key, data)
        VALUES %s
        ON CONFLICT (key) DO UPDATE
        SET data = EXCLUDED.data
    """
    )

    values = [(key, Json(model.model_dump())) for key, model in keys_models]
    execute_values(cur, query, values, template="(%s, %s)")


def load_models(cur, cls):
    cur.execute("SELECT key, data FROM metadata_store")
    rows = cur.fetchall()
    return [TypeAdapter(cls).validate_python(row["data"]) for row in rows]


class PGVectorIndex(EmbeddingIndex):
    def __init__(self, vector_db: VectorDB, dimension: int, cursor):
        self.cursor = cursor
        self.table_name = f"vector_store_{vector_db.identifier}"

        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                document JSONB,
                embedding vector({dimension})
            )
        """
        )

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        values = []
        for i, chunk in enumerate(chunks):
            values.append(
                (
                    f"{chunk.document_id}:chunk-{i}",
                    Json(chunk.model_dump()),
                    embeddings[i].tolist(),
                )
            )

        query = sql.SQL(
            f"""
        INSERT INTO {self.table_name} (id, document, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, document = EXCLUDED.document
    """
        )
        execute_values(self.cursor, query, values, template="(%s, %s, %s::vector)")

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryChunksResponse:
        self.cursor.execute(
            f"""
        SELECT document, embedding <-> %s::vector AS distance
        FROM {self.table_name}
        ORDER BY distance
        LIMIT %s
    """,
            (embedding.tolist(), k),
        )
        results = self.cursor.fetchall()

        chunks = []
        scores = []
        for doc, dist in results:
            chunks.append(Chunk(**doc))
            scores.append(1.0 / float(dist))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")


class PGVectorVectorDBAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: PGVectorConfig, inference_api: Api.inference) -> None:
        self.config = config
        self.inference_api = inference_api
        self.cursor = None
        self.conn = None
        self.cache = {}

    async def initialize(self) -> None:
        log.info(f"Initializing PGVector memory adapter with config: {self.config}")
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            version = check_extension_version(self.cursor)
            if version:
                log.info(f"Vector extension version: {version}")
            else:
                raise RuntimeError("Vector extension is not installed.")

            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_store (
                    key TEXT PRIMARY KEY,
                    data JSONB
                )
            """
            )
        except Exception as e:
            log.exception("Could not connect to PGVector database server")
            raise RuntimeError("Could not connect to PGVector database server") from e

    async def shutdown(self) -> None:
        pass

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        upsert_models(self.cursor, [(vector_db.identifier, vector_db)])

        index = PGVectorIndex(vector_db, vector_db.embedding_dimension, self.cursor)
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db, index, self.inference_api
        )

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        return await index.query_chunks(query, params)

    async def _get_and_cache_vector_db_index(
        self, vector_db_id: str
    ) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        index = PGVectorIndex(vector_db, vector_db.embedding_dimension, self.cursor)
        self.cache[vector_db_id] = VectorDBWithIndex(
            vector_db, index, self.inference_api
        )
        return self.cache[vector_db_id]
