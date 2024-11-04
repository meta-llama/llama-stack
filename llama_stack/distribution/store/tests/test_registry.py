# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from llama_stack.distribution.store import *  # noqa: F403
from llama_stack.apis.memory_banks import VectorMemoryBankDef
from llama_stack.providers.utils.kvstore import kvstore_impl, SqliteKVStoreConfig
from llama_stack.distribution.datatypes import *  # noqa: F403


@pytest.mark.asyncio
async def test_registry():
    config = SqliteKVStoreConfig(db_path="/tmp/test_registry.db")
    # delete the file if it exists
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    registry = DiskDistributionRegistry(await kvstore_impl(config))
    bank = VectorMemoryBankDef(
        identifier="test_bank",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
        provider_id="bar",
    )
    model = ModelDefWithProvider(
        identifier="test_model",
        llama_model="Llama3.2-3B-Instruct",
        provider_id="foo",
    )

    await registry.register(bank)
    await registry.register(model)
    results = await registry.get("test_bank")
    assert len(results) == 1
    result_bank = results[0]
    assert result_bank.identifier == bank.identifier
    assert result_bank.embedding_model == bank.embedding_model
    assert result_bank.chunk_size_in_tokens == bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == bank.overlap_size_in_tokens
    assert result_bank.provider_id == bank.provider_id

    results = await registry.get("test_model")
    assert len(results) == 1
    result_model = results[0]
    assert result_model.identifier == model.identifier
    assert result_model.llama_model == model.llama_model
    assert result_model.provider_id == model.provider_id
