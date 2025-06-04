# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import hashlib
import json
import logging
import sqlite3
import struct
import uuid
from typing import Any

import numpy as np
import sqlite_vec
from numpy.typing import NDArray

from llama_stack.apis.files.files import Files
from llama_stack.apis.inference.inference import Inference
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_RRF,
    RERANKER_TYPE_WEIGHTED,
    EmbeddingIndex,
    VectorDBWithIndex,
)

logger = logging.getLogger(__name__)

# Specifying search mode is dependent on the VectorIO provider.
VECTOR_SEARCH = "vector"
KEYWORD_SEARCH = "keyword"
HYBRID_SEARCH = "hybrid"
SEARCH_MODES = {VECTOR_SEARCH, KEYWORD_SEARCH, HYBRID_SEARCH}


def serialize_vector(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary representation."""
    return struct.pack(f"{len(vector)}f", *vector)


def _create_sqlite_connection(db_path):
    """Create a SQLite connection with sqlite_vec extension loaded."""
    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0,1] range using min-max normalization."""
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score
    if score_range > 0:
        return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}
    return {doc_id: 1.0 for doc_id in scores}


def _weighted_rerank(
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """ReRanker that uses weighted average of scores."""
    all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    normalized_vector_scores = _normalize_scores(vector_scores)
    normalized_keyword_scores = _normalize_scores(keyword_scores)

    return {
        doc_id: (alpha * normalized_keyword_scores.get(doc_id, 0.0))
        + ((1 - alpha) * normalized_vector_scores.get(doc_id, 0.0))
        for doc_id in all_ids
    }


def _rrf_rerank(
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    impact_factor: float = 60.0,
) -> dict[str, float]:
    """ReRanker that uses Reciprocal Rank Fusion."""
    # Convert scores to ranks
    vector_ranks = {
        doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted(vector_scores.items(), key=lambda x: x[1], reverse=True))
    }
    keyword_ranks = {
        doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True))
    }

    all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    rrf_scores = {}
    for doc_id in all_ids:
        vector_rank = vector_ranks.get(doc_id, float("inf"))
        keyword_rank = keyword_ranks.get(doc_id, float("inf"))
        # RRF formula: score = 1/(k + r) where k is impact_factor and r is the rank
        rrf_scores[doc_id] = (1.0 / (impact_factor + vector_rank)) + (1.0 / (impact_factor + keyword_rank))
    return rrf_scores


class SQLiteVecIndex(EmbeddingIndex):
    """
    An index implementation that stores embeddings in a SQLite virtual table using sqlite-vec.
    Two tables are used:
      - A metadata table (chunks_{bank_id}) that holds the chunk JSON.
      - A virtual table (vec_chunks_{bank_id}) that holds the serialized vector.
      - An FTS5 table (fts_chunks_{bank_id}) for full-text keyword search.
    """

    def __init__(self, dimension: int, db_path: str, bank_id: str):
        self.dimension = dimension
        self.db_path = db_path
        self.bank_id = bank_id
        self.metadata_table = f"chunks_{bank_id}".replace("-", "_")
        self.vector_table = f"vec_chunks_{bank_id}".replace("-", "_")
        self.fts_table = f"fts_chunks_{bank_id}".replace("-", "_")

    @classmethod
    async def create(cls, dimension: int, db_path: str, bank_id: str):
        instance = cls(dimension, db_path, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        def _init_tables():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
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
                connection.commit()
                # FTS5 table (for keyword search) - creating both the tables by default. Will use the relevant one
                # based on query. Implementation of the change on client side will allow passing the search_mode option
                # during initialization to make it easier to create the table that is required.
                cur.execute(f"""
                            CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table}
                            USING fts5(id, content);
                        """)
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_init_tables)

    async def delete(self) -> None:
        def _drop_tables():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                cur.execute(f"DROP TABLE IF EXISTS {self.metadata_table};")
                cur.execute(f"DROP TABLE IF EXISTS {self.vector_table};")
                cur.execute(f"DROP TABLE IF EXISTS {self.fts_table};")
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_drop_tables)

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray, batch_size: int = 500):
        """
        Add new chunks along with their embeddings using batch inserts.
        For each chunk, we insert its JSON into the metadata table and then insert its
        embedding (serialized to raw bytes) into the virtual table using the assigned rowid.
        If any insert fails, the transaction is rolled back to maintain consistency.
        Also inserts chunk content into FTS table for keyword search support.
        """
        assert all(isinstance(chunk.content, str) for chunk in chunks), "SQLiteVecIndex only supports text chunks"

        def _execute_all_batch_inserts():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()

            try:
                cur.execute("BEGIN TRANSACTION")
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    batch_embeddings = embeddings[i : i + batch_size]

                    # Insert metadata
                    metadata_data = [
                        (generate_chunk_id(chunk.metadata["document_id"], chunk.content), chunk.model_dump_json())
                        for chunk in batch_chunks
                    ]
                    cur.executemany(
                        f"""
                        INSERT INTO {self.metadata_table} (id, chunk)
                        VALUES (?, ?)
                        ON CONFLICT(id) DO UPDATE SET chunk = excluded.chunk;
                        """,
                        metadata_data,
                    )

                    # Insert vector embeddings
                    embedding_data = [
                        (
                            (
                                generate_chunk_id(chunk.metadata["document_id"], chunk.content),
                                serialize_vector(emb.tolist()),
                            )
                        )
                        for chunk, emb in zip(batch_chunks, batch_embeddings, strict=True)
                    ]
                    cur.executemany(
                        f"INSERT INTO {self.vector_table} (id, embedding) VALUES (?, ?);",
                        embedding_data,
                    )

                    # Insert FTS content
                    fts_data = [
                        (generate_chunk_id(chunk.metadata["document_id"], chunk.content), chunk.content)
                        for chunk in batch_chunks
                    ]
                    # DELETE existing entries with same IDs (FTS5 doesn't support ON CONFLICT)
                    cur.executemany(
                        f"DELETE FROM {self.fts_table} WHERE id = ?;",
                        [(row[0],) for row in fts_data],
                    )

                    # INSERT new entries
                    cur.executemany(
                        f"INSERT INTO {self.fts_table} (id, content) VALUES (?, ?);",
                        fts_data,
                    )

                connection.commit()

            except sqlite3.Error as e:
                connection.rollback()
                logger.error(f"Error inserting into {self.vector_table}: {e}")
                raise

            finally:
                cur.close()
                connection.close()

        # Run batch insertion in a background thread
        await asyncio.to_thread(_execute_all_batch_inserts)

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Performs vector-based search using a virtual table for vector similarity.
        """

        def _execute_query():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
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
                return cur.fetchall()
            finally:
                cur.close()
                connection.close()

        rows = await asyncio.to_thread(_execute_query)
        chunks, scores = [], []
        for row in rows:
            _id, chunk_json, distance = row
            score = 1.0 / distance if distance != 0 else float("inf")
            if score < score_threshold:
                continue
            try:
                chunk = Chunk.model_validate_json(chunk_json)
            except Exception as e:
                logger.error(f"Error parsing chunk JSON for id {_id}: {e}")
                continue
            chunks.append(chunk)
            scores.append(score)
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Performs keyword-based search using SQLite FTS5 for relevance-ranked full-text search.
        """

        def _execute_query():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                query_sql = f"""
                    SELECT DISTINCT m.id, m.chunk, bm25({self.fts_table}) AS score
                    FROM {self.fts_table} AS f
                    JOIN {self.metadata_table} AS m ON m.id = f.id
                    WHERE f.content MATCH ?
                    ORDER BY score ASC
                    LIMIT ?;
                """
                cur.execute(query_sql, (query_string, k))
                return cur.fetchall()
            finally:
                cur.close()
                connection.close()

        rows = await asyncio.to_thread(_execute_query)
        chunks, scores = [], []
        for row in rows:
            _id, chunk_json, score = row
            # BM25 scores returned by sqlite-vec are NEGATED (i.e., more relevant = more negative).
            # This design is intentional to simplify sorting by ascending score.
            # Reference: https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html
            if score > -score_threshold:
                continue
            try:
                chunk = Chunk.model_validate_json(chunk_json)
            except Exception as e:
                logger.error(f"Error parsing chunk JSON for id {_id}: {e}")
                continue
            chunks.append(chunk)
            scores.append(score)
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str = RERANKER_TYPE_RRF,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search using a configurable re-ranking strategy.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker

        Returns:
            QueryChunksResponse with combined results
        """
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        # Convert responses to score dictionaries using generate_chunk_id
        vector_scores = {
            generate_chunk_id(chunk.metadata["document_id"], str(chunk.content)): score
            for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            generate_chunk_id(chunk.metadata["document_id"], str(chunk.content)): score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the specified reranker
        if reranker_type == RERANKER_TYPE_WEIGHTED:
            alpha = reranker_params.get("alpha", 0.5)
            combined_scores = _weighted_rerank(vector_scores, keyword_scores, alpha)
        else:
            # Default to RRF for None, RRF, or any unknown types
            impact_factor = reranker_params.get("impact_factor", 60.0)
            combined_scores = _rrf_rerank(vector_scores, keyword_scores, impact_factor)

        # Sort by combined score and get top k results
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create a map of chunk_id to chunk for both responses
        chunk_map = {}
        for c in vector_response.chunks:
            chunk_id = generate_chunk_id(c.metadata["document_id"], str(c.content))
            chunk_map[chunk_id] = c
        for c in keyword_response.chunks:
            chunk_id = generate_chunk_id(c.metadata["document_id"], str(c.content))
            chunk_map[chunk_id] = c

        # Use the map to look up chunks by their IDs
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)


class SQLiteVecVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    """
    A VectorIO implementation using SQLite + sqlite_vec.
    This class handles vector database registration (with metadata stored in a table named `vector_dbs`)
    and creates a cache of VectorDBWithIndex instances (each wrapping a SQLiteVecIndex).
    """

    def __init__(self, config, inference_api: Inference, files_api: Files | None) -> None:
        self.config = config
        self.inference_api = inference_api
        self.files_api = files_api
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        def _setup_connection():
            # Open a connection to the SQLite database (the file is specified in the config).
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                # Create a table to persist vector DB registrations.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vector_dbs (
                        id TEXT PRIMARY KEY,
                        metadata TEXT
                    );
                """)
                # Create a table to persist OpenAI vector stores.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS openai_vector_stores (
                        id TEXT PRIMARY KEY,
                        metadata TEXT
                    );
                """)
                connection.commit()
                # Load any existing vector DB registrations.
                cur.execute("SELECT metadata FROM vector_dbs")
                vector_db_rows = cur.fetchall()
                return vector_db_rows
            finally:
                cur.close()
                connection.close()

        vector_db_rows = await asyncio.to_thread(_setup_connection)

        # Load existing vector DBs
        for row in vector_db_rows:
            vector_db_data = row[0]
            vector_db = VectorDB.model_validate_json(vector_db_data)
            index = await SQLiteVecIndex.create(
                vector_db.embedding_dimension,
                self.config.db_path,
                vector_db.identifier,
            )
            self.cache[vector_db.identifier] = VectorDBWithIndex(vector_db, index, self.inference_api)

        # Load existing OpenAI vector stores using the mixin method
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        # nothing to do since we don't maintain a persistent connection
        pass

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        def _register_db():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO vector_dbs (id, metadata) VALUES (?, ?)",
                    (vector_db.identifier, vector_db.model_dump_json()),
                )
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_register_db)
        index = await SQLiteVecIndex.create(
            vector_db.embedding_dimension,
            self.config.db_path,
            vector_db.identifier,
        )
        self.cache[vector_db.identifier] = VectorDBWithIndex(vector_db, index, self.inference_api)

    async def list_vector_dbs(self) -> list[VectorDB]:
        return [v.vector_db for v in self.cache.values()]

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if vector_db_id not in self.cache:
            logger.warning(f"Vector DB {vector_db_id} not found")
            return
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

        def _delete_vector_db_from_registry():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute("DELETE FROM vector_dbs WHERE id = ?", (vector_db_id,))
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_delete_vector_db_from_registry)

    # OpenAI Vector Store Mixin abstract method implementations
    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Save vector store metadata to SQLite database."""

        def _store():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO openai_vector_stores (id, metadata) VALUES (?, ?)",
                    (store_id, json.dumps(store_info)),
                )
                connection.commit()
            except Exception as e:
                logger.error(f"Error saving openai vector store {store_id}: {e}")
                raise
            finally:
                cur.close()
                connection.close()

        try:
            await asyncio.to_thread(_store)
        except Exception as e:
            logger.error(f"Error saving openai vector store {store_id}: {e}")
            raise

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from SQLite database."""

        def _load():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute("SELECT metadata FROM openai_vector_stores")
                rows = cur.fetchall()
                return rows
            finally:
                cur.close()
                connection.close()

        rows = await asyncio.to_thread(_load)
        stores = {}
        for row in rows:
            store_data = row[0]
            store_info = json.loads(store_data)
            stores[store_info["id"]] = store_info
        return stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in SQLite database."""

        def _update():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute(
                    "UPDATE openai_vector_stores SET metadata = ? WHERE id = ?",
                    (json.dumps(store_info), store_id),
                )
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_update)

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from SQLite database."""

        def _delete():
            connection = _create_sqlite_connection(self.config.db_path)
            cur = connection.cursor()
            try:
                cur.execute("DELETE FROM openai_vector_stores WHERE id = ?", (store_id,))
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_delete)

    async def insert_chunks(self, vector_db_id: str, chunks: list[Chunk], ttl_seconds: int | None = None) -> None:
        if vector_db_id not in self.cache:
            raise ValueError(f"Vector DB {vector_db_id} not found. Found: {list(self.cache.keys())}")
        # The VectorDBWithIndex helper is expected to compute embeddings via the inference_api
        # and then call our index's add_chunks.
        await self.cache[vector_db_id].insert_chunks(chunks)

    async def query_chunks(
        self, vector_db_id: str, query: Any, params: dict[str, Any] | None = None
    ) -> QueryChunksResponse:
        if vector_db_id not in self.cache:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        return await self.cache[vector_db_id].query_chunks(query, params)


def generate_chunk_id(document_id: str, chunk_text: str) -> str:
    """Generate a unique chunk ID using a hash of document ID and chunk text."""
    hash_input = f"{document_id}:{chunk_text}".encode()
    return str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))
