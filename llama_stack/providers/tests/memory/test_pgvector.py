# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from typing import List, Tuple
from unittest import mock
from unittest.mock import patch

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extras import execute_values, Json
from pydantic import BaseModel

from llama_stack.apis.memory import MemoryBankType, VectorMemoryBank
from llama_stack.providers.remote.memory.pgvector.config import PGVectorConfig
from llama_stack.providers.remote.memory.pgvector.pgvector import (
    PGVectorIndex,
    PGVectorMemoryAdapter,
)
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.memory.vector_store import (
    ALL_MINILM_L6_V2_DIMENSION,
    BankWithIndex,
)


TEST_MEMORY_BANKS_PREFIX = "test_memory_banks:"


@mock.patch("psycopg2.connect")
async def _noop_pgvectormemoryadapter_initialize(self, mock_connect):
    print("Running _noop_pgvectormemoryadapter_initialize()")

    try:
        self.conn = psycopg2.connect(client_encoding="utf8")
        self.conn.autocommit = True
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # self.cursor.connection.set_client_encoding({"encoding": "UTF8"})

        print(f"cursor: {self.cursor}")
        print(f"connection: {self.cursor.connection}")
        print(f"encoding: {self.cursor.connection.encoding}")

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

    self.kvstore = await kvstore_impl(self.config.kvstore)
    # Load existing banks from kvstore
    start_key = TEST_MEMORY_BANKS_PREFIX
    end_key = f"{TEST_MEMORY_BANKS_PREFIX}\xff"
    stored_banks = await self.kvstore.range(start_key, end_key)

    for bank_data in stored_banks:
        bank = VectorMemoryBank.model_validate_json(bank_data)
        index = BankWithIndex(
            bank=bank, index=PGVectorIndex(ALL_MINILM_L6_V2_DIMENSION)
        )
        self.cache[bank.identifier] = index


@mock.patch("psycopg2.connect")
def _noop_upsert_models(cur, keys_models: List[Tuple[str, BaseModel]], mock_connect):
    print("Running _noop_upsert_models()")
    conn = psycopg2.connect("")
    conn.autocommit = True
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.connection.set_client_encoding("UTF8")

    print(f"cursor: {cursor}")
    print(f"connection: {cursor.connection}")
    print(f"encoding: {cursor.connection.encoding}")

    query = sql.SQL(
        """
        INSERT INTO metadata_store (key, data)
        VALUES %s
        ON CONFLICT (key) DO UPDATE
        SET data = EXCLUDED.data
    """
    )

    values = [(key, Json(model.dict())) for key, model in keys_models]
    execute_values(cursor, query, values, template="(%s, %s)")


@patch.object(
    PGVectorMemoryAdapter, "initialize", _noop_pgvectormemoryadapter_initialize
)
# @patch("llama_stack.providers.remote.memory.pgvector.pgvector.upsert_models", _noop_upsert_models)
class TestPGVectorMemoryAdapter:
    @pytest.fixture
    def pgvector_memory_adapter(self):
        # Create a temporary SQLite database file
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        config = PGVectorConfig(kvstore=SqliteKVStoreConfig(db_path=temp_db.name))
        return PGVectorMemoryAdapter(config)

    @pytest.mark.asyncio
    async def test_initialize(self, pgvector_memory_adapter):
        # Test empty initialization
        await pgvector_memory_adapter.initialize()
        assert len(pgvector_memory_adapter.cache) == 0

        # Test initialization with existing banks
        bank = VectorMemoryBank(
            identifier="test_bank",
            provider_id="",
            memory_bank_type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )

        # Register a bank and reinitialize to test loading
        await pgvector_memory_adapter.register_memory_bank(bank)

        # Create new instance to test initialization with existing data
        new_mem_adpt = PGVectorMemoryAdapter(pgvector_memory_adapter.config)
        await new_mem_adpt.initialize()

        assert len(new_mem_adpt.cache) == 1
        assert "test_bank" in new_mem_adpt.cache

    @pytest.mark.asyncio
    async def test_register_memory_bank(self, pgvector_memory_adapter):
        bank = VectorMemoryBank(
            identifier="test_bank",
            provider_id="",
            memory_bank_type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )

        await pgvector_memory_adapter.initialize()
        await pgvector_memory_adapter.register_memory_bank(bank)

        assert "test_bank" in pgvector_memory_adapter.cache
        assert pgvector_memory_adapter.cache["test_bank"].bank == bank

        # Verify persistence
        new_mem_adpt = PGVectorMemoryAdapter(pgvector_memory_adapter.config)
        await new_mem_adpt.initialize()
        assert "test_bank" in new_mem_adpt.cache


if __name__ == "__main__":
    pytest.main([__file__])
