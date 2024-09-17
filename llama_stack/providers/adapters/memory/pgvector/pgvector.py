# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from typing import List, Tuple

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import execute_values, Json
from pydantic import BaseModel
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


def upsert_models(cur, keys_models: List[Tuple[str, BaseModel]]):
    query = sql.SQL(
        """
        INSERT INTO metadata_store (key, data)
        VALUES %s
        ON CONFLICT (key) DO UPDATE
        SET data = EXCLUDED.data
    """
    )

    values = [(key, Json(model.dict())) for key, model in keys_models]
    execute_values(cur, query, values, template="(%s, %s)")


def load_models(cur, keys: List[str], cls):
    query = "SELECT key, data FROM metadata_store"
    if keys:
        placeholders = ",".join(["%s"] * len(keys))
        query += f" WHERE key IN ({placeholders})"
        cur.execute(query, keys)
    else:
        cur.execute(query)

    rows = cur.fetchall()
    return [cls(**row["data"]) for row in rows]


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
            print(f"Adding chunk #{i} tokens={chunk.token_count}")
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

            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_store (
                    key TEXT PRIMARY KEY,
                    data JSONB
                )
            """
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError("Could not connect to PGVector database server") from e

    async def shutdown(self) -> None:
        pass

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        bank_id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=bank_id,
            name=name,
            config=config,
            url=url,
        )
        upsert_models(
            self.cursor,
            [
                (bank.bank_id, bank),
            ],
        )
        index = BankWithIndex(
            bank=bank,
            index=PGVectorIndex(bank, ALL_MINILM_L6_V2_DIMENSION, self.cursor),
        )
        self.cache[bank_id] = index
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        bank_index = await self._get_and_cache_bank_index(bank_id)
        if bank_index is None:
            return None
        return bank_index.bank

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        banks = load_models(self.cursor, [bank_id], MemoryBank)
        if not banks:
            return None

        bank = banks[0]
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
