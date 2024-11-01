import pytest
import pytest_asyncio
from llama_stack.distribution.store import *
from llama_stack.apis.memory_banks import GraphMemoryBankDef, VectorMemoryBankDef
from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.providers.utils.kvstore import kvstore_impl, SqliteKVStoreConfig
from llama_stack.distribution.datatypes import *  # noqa: F403


@pytest.mark.asyncio
async def test_registry():
    registry = DiskRegistry(await kvstore_impl(SqliteKVStoreConfig()))
    bank = VectorMemoryBankDef(
        identifier="test_bank",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
        provider_id="bar",
    )

    await registry.register(bank)
    result_bank = await registry.get("test_bank")
    # assert result_bank == bank
    assert result_bank.identifier == bank.identifier
    assert result_bank.embedding_model == bank.embedding_model
    assert result_bank.chunk_size_in_tokens == bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == bank.overlap_size_in_tokens
    assert result_bank.provider_id == bank.provider_id
