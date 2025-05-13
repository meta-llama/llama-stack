# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator

import pytest

from llama_stack.providers.remote.files.object.s3.config import S3FilesImplConfig
from llama_stack.providers.remote.files.object.s3.s3_files import S3FilesAdapter
from llama_stack.providers.utils.kvstore import KVStore, kvstore_impl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
def s3_config():
    """Create S3 configuration for MinIO."""
    return S3FilesImplConfig(
        aws_access_key_id="ROOTNAME",
        aws_secret_access_key="CHANGEME123",
        region_name="us-east-1",
        endpoint_url="http://localhost:9000",
    )


@pytest.fixture
async def kvstore() -> AsyncGenerator[KVStore, None]:
    """Create a SQLite KV store for testing."""
    config = SqliteKVStoreConfig(
        path=":memory:"  # Use in-memory SQLite for tests
    )
    store = await kvstore_impl(config)
    await store.initialize()
    yield store


@pytest.fixture
async def s3_files(s3_config, kvstore) -> AsyncGenerator[S3FilesAdapter, None]:
    """Create S3FilesAdapter instance for testing."""
    adapter = S3FilesAdapter(s3_config, kvstore)
    await adapter.initialize()
    yield adapter
