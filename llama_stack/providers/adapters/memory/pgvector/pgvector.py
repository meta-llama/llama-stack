# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import execute_values, Json

from llama_stack.apis.memory import *  # noqa: F403

from llama_stack.providers.utils.memory.vector_store import (
    ALL_MINILM_L6_V2_DIMENSION,
    BankWithIndex,
    EmbeddingIndex,
)

from .config import PGVectorConfig


def check_extension_version(cur):
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    return result[0] if result else None


class PGVectorIndex(EmbeddingIndex):
    def __init__(self, bank: MemoryBank, dimension: int, cursor):
        self.cursor = cursor
        self.table_name = f"vector_store_{bank.name}"

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
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        values = []
        for i, chunk in enumerate(chunks):
            values.append(
                (
                    f"{chunk.document_id}:chunk-{i}",
                    Json(chunk.dict()),
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

    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
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

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class PGVectorMemoryAdapter(Memory):
    def __init__(self, config: PGVectorConfig) -> None:
        print(f"Initializing PGVectorMemoryAdapter -> {config.host}:{config.port}")
        self.config = config
        self.cursor = None
        self.conn = None
        self.cache = {}

    async def initialize(self) -> None:
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
            )
            self.cursor = self.conn.cursor()

            version = check_extension_version(self.cursor)
            if version:
                print(f"Vector extension version: {version}")
            else:
                raise RuntimeError("Vector extension is not installed.")

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError("Could not connect to PGVector database server") from e

    async def shutdown(self) -> None:
        pass

    async def register_memory_bank(
        self,
        memory_bank: MemoryBankDef,
    ) -> None:
        assert (
            memory_bank.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.type}"

        index = BankWithIndex(
            bank=memory_bank,
            index=PGVectorIndex(memory_bank, ALL_MINILM_L6_V2_DIMENSION, self.cursor),
        )
        self.cache[bank_id] = index

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found")

        index = BankWithIndex(
            bank=bank,
            index=PGVectorIndex(bank, ALL_MINILM_L6_V2_DIMENSION, self.cursor),
        )
        self.cache[bank_id] = index
        return index

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        return await index.query_documents(query, params)
