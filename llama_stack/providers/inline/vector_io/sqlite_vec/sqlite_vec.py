# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import logging
import sqlite3
import struct
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import sqlite_vec
from numpy.typing import NDArray

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import EmbeddingIndex, VectorDBWithIndex

logger = logging.getLogger(__name__)


def serialize_vector(vector: List[float]) -> bytes:
    """Serialize a list of floats into a compact binary representation."""
    return struct.pack(f"{len(vector)}f", *vector)


class SQLiteVecIndex(EmbeddingIndex):
    """
    An index implementation that stores embeddings in a SQLite virtual table using sqlite-vec.
    Two tables are used:
      - A metadata table (chunks_{bank_id}) that holds the chunk JSON.
      - A virtual table (vec_chunks_{bank_id}) that holds the serialized vector.
      - An FTS5 table (fts_chunks_{bank_id}) for full-text keyword search.
    """

    def __init__(self, dimension: int, connection: sqlite3.Connection, bank_id: str):
        self.dimension = dimension
        self.connection = connection
        self.bank_id = bank_id
        self.metadata_table = f"chunks_{bank_id}".replace("-", "_")
        self.vector_table = f"vec_chunks_{bank_id}".replace("-", "_")
        self.fts_table = f"fts_chunks_{bank_id}".replace("-", "_")

    @classmethod
    async def create(cls, dimension: int, connection: sqlite3.Connection, bank_id: str):
        instance = cls(dimension, connection, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        cur = self.connection.cursor()
        # Create the table to store chunk metadata.
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                id TEXT PRIMARY KEY,
                chunk TEXT
            );
        """)
        # Create the virtual table for embeddings.
        cur.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.vector_table}
            USING vec0(embedding FLOAT[{self.dimension}], id TEXT);
        """)
        # FTS5 table (for keyword search) - creating both the tables by default. Will use the relevant one based on
        # query.
        cur.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table}
                    USING fts5(id, content);
                """)
        self.connection.commit()

    async def delete(self):
        cur = self.connection.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {self.metadata_table};")
        cur.execute(f"DROP TABLE IF EXISTS {self.vector_table};")
        cur.execute(f"DROP TABLE IF EXISTS {self.fts_table};")
        self.connection.commit()

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray, batch_size: int = 500):
        """
        Add new chunks along with their embeddings using batch inserts.
        For each chunk, we insert its JSON into the metadata table and then insert its
        embedding (serialized to raw bytes) into the virtual table using the assigned rowid.
        If any insert fails, the transaction is rolled back to maintain consistency.
        """
        cur = self.connection.cursor()
        try:
            # Start transaction
            cur.execute("BEGIN TRANSACTION")
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                # Prepare metadata inserts
                metadata_data = [
                    (generate_chunk_id(chunk.metadata["document_id"], chunk.content), chunk.model_dump_json())
                    for chunk in batch_chunks
                ]
                # Insert metadata (ON CONFLICT to avoid duplicates)
                cur.executemany(
                    f"""
                    INSERT INTO {self.metadata_table} (id, chunk)
                    VALUES (?, ?)
                    ON CONFLICT(id) DO UPDATE SET chunk = excluded.chunk;
                    """,
                    metadata_data,
                )
                # Prepare embeddings inserts
                embedding_data = [
                    (generate_chunk_id(chunk.metadata["document_id"], chunk.content), serialize_vector(emb.tolist()))
                    for chunk, emb in zip(batch_chunks, batch_embeddings, strict=True)
                ]
                # Insert embeddings in batch
                cur.executemany(f"INSERT INTO {self.vector_table} (id, embedding) VALUES (?, ?);", embedding_data)

                fts_data = [
                    (generate_chunk_id(chunk.metadata["document_id"], chunk.content), chunk.content)
                    for chunk in batch_chunks
                ]
                cur.executemany(f"INSERT INTO {self.fts_table} (id, content) VALUES (?, ?);", fts_data)
            self.connection.commit()

        except sqlite3.Error as e:
            self.connection.rollback()  # Rollback on failure
            logger.error(f"Error inserting into {self.vector_table}: {e}")

        finally:
            cur.close()  # Ensure cursor is closed

    async def query(
        self, embedding: Optional[NDArray], k: int, score_threshold: float, query_str: Optional[str], search_mode: str
    ) -> QueryChunksResponse:
        """
        Supports both vector-based and keyword-based searches.

        1. Vector Search (`search_mode`="vector")
        Query for the k most similar chunks. We convert the query embedding to a blob and run a SQL query
        against the virtual table. The SQL joins the metadata table to recover the chunk JSON.

        2. Keyword search (`search_mode`="keyword")
        Uses SQLite's `FTS5` module to find matches for `query_str` and ranks results using BM25 relevance scoring.
        Note: BM25/FTS is not guaranteed to return a result during retrieval.
        """
        cur = self.connection.cursor()
        if search_mode == "vector":
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            emb_blob = serialize_vector(emb_list)

            query_sql = f"""
                SELECT m.id, m.chunk, v.distance
                FROM {self.vector_table} AS v
                JOIN {self.metadata_table} AS m ON m.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance;
            """
            cur.execute(query_sql, (emb_blob, k))
        elif search_mode == "keyword":
            if query_str is None:
                raise ValueError("query_str is required for keyword search.")
            # Full-Text Search SQL
            query_sql = f"""
                SELECT DISTINCT m.id, m.chunk, bm25({self.fts_table}) AS score
                FROM {self.fts_table} AS f
                JOIN {self.metadata_table} AS m ON m.id = f.id
                WHERE f.content MATCH ?
                ORDER BY score ASC
                LIMIT ?;
            """
            cur.execute(query_sql, (query_str, k))
        else:
            raise ValueError(f"Invalid search_mode: {search_mode}")

        rows = cur.fetchall()
        cur.close()

        chunks = []
        scores = []
        for row in rows:
            if search_mode == "vector":
                _id, chunk_json, distance = row
                score = 1.0 / distance if distance != 0 else float("inf")  # Convert distance to score
            else:
                _id, chunk_json, score = row

            try:
                chunk = Chunk.model_validate_json(chunk_json)
            except Exception as e:
                logger.error(f"Error parsing chunk JSON for id {_id}: {e}")
                continue

            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)


class SQLiteVecVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    """
    A VectorIO implementation using SQLite + sqlite_vec.
    This class handles vector database registration (with metadata stored in a table named `vector_dbs`)
    and creates a cache of VectorDBWithIndex instances (each wrapping a SQLiteVecIndex).
    """

    def __init__(self, config, inference_api: Api.inference) -> None:
        self.config = config
        self.inference_api = inference_api
        self.cache: Dict[str, VectorDBWithIndex] = {}
        self.connection: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        # Open a connection to the SQLite database (the file is specified in the config).
        self.connection = sqlite3.connect(self.config.db_path)
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        cur = self.connection.cursor()
        # Create a table to persist vector DB registrations.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vector_dbs (
                id TEXT PRIMARY KEY,
                metadata TEXT
            );
        """)
        self.connection.commit()
        # Load any existing vector DB registrations.
        cur.execute("SELECT metadata FROM vector_dbs")
        rows = cur.fetchall()
        for row in rows:
            vector_db_data = row[0]
            vector_db = VectorDB.model_validate_json(vector_db_data)
            index = await SQLiteVecIndex.create(vector_db.embedding_dimension, self.connection, vector_db.identifier)
            self.cache[vector_db.identifier] = VectorDBWithIndex(vector_db, index, self.inference_api)

    async def shutdown(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        if self.connection is None:
            raise RuntimeError("SQLite connection not initialized")
        cur = self.connection.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO vector_dbs (id, metadata) VALUES (?, ?)",
            (vector_db.identifier, vector_db.model_dump_json()),
        )
        self.connection.commit()
        index = await SQLiteVecIndex.create(vector_db.embedding_dimension, self.connection, vector_db.identifier)
        self.cache[vector_db.identifier] = VectorDBWithIndex(vector_db, index, self.inference_api)

    async def list_vector_dbs(self) -> List[VectorDB]:
        return [v.vector_db for v in self.cache.values()]

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if self.connection is None:
            raise RuntimeError("SQLite connection not initialized")
        if vector_db_id not in self.cache:
            logger.warning(f"Vector DB {vector_db_id} not found")
            return
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]
        cur = self.connection.cursor()
        cur.execute("DELETE FROM vector_dbs WHERE id = ?", (vector_db_id,))
        self.connection.commit()

    async def insert_chunks(self, vector_db_id: str, chunks: List[Chunk], ttl_seconds: Optional[int] = None) -> None:
        if vector_db_id not in self.cache:
            raise ValueError(f"Vector DB {vector_db_id} not found. Found: {list(self.cache.keys())}")
        # The VectorDBWithIndex helper is expected to compute embeddings via the inference_api
        # and then call our index's add_chunks.
        await self.cache[vector_db_id].insert_chunks(chunks)

    async def query_chunks(
        self, vector_db_id: str, query: Any, params: Optional[Dict[str, Any]] = None
    ) -> QueryChunksResponse:
        if vector_db_id not in self.cache:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        return await self.cache[vector_db_id].query_chunks(query, params)


def generate_chunk_id(document_id: str, chunk_text: str) -> str:
    """Generate a unique chunk ID using a hash of document ID and chunk text."""
    hash_input = f"{document_id}:{chunk_text}".encode("utf-8")
    return str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))
