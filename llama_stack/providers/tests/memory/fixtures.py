# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile

import pytest
import pytest_asyncio

from llama_stack.apis.inference import ModelInput, ModelType

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.memory.chroma import ChromaInlineImplConfig
from llama_stack.providers.inline.memory.faiss import FaissImplConfig
from llama_stack.providers.remote.memory.chroma import ChromaRemoteImplConfig
from llama_stack.providers.remote.memory.pgvector import PGVectorConfig
from llama_stack.providers.remote.memory.weaviate import WeaviateConfig
from llama_stack.providers.tests.resolver import construct_stack_for_test
from llama_stack.providers.utils.kvstore import SqliteKVStoreConfig
from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def embedding_model(request):
    if hasattr(request, "param"):
        return request.param
    return request.config.getoption("--embedding-model", None)


@pytest.fixture(scope="session")
def memory_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def memory_faiss() -> ProviderFixture:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="faiss",
                provider_type="inline::faiss",
                config=FaissImplConfig(
                    kvstore=SqliteKVStoreConfig(db_path=temp_file.name).model_dump(),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def memory_pgvector() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="pgvector",
                provider_type="remote::pgvector",
                config=PGVectorConfig(
                    host=os.getenv("PGVECTOR_HOST", "localhost"),
                    port=os.getenv("PGVECTOR_PORT", 5432),
                    db=get_env_or_fail("PGVECTOR_DB"),
                    user=get_env_or_fail("PGVECTOR_USER"),
                    password=get_env_or_fail("PGVECTOR_PASSWORD"),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def memory_weaviate() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="weaviate",
                provider_type="remote::weaviate",
                config=WeaviateConfig().model_dump(),
            )
        ],
        provider_data=dict(
            weaviate_api_key=get_env_or_fail("WEAVIATE_API_KEY"),
            weaviate_cluster_url=get_env_or_fail("WEAVIATE_CLUSTER_URL"),
        ),
    )


@pytest.fixture(scope="session")
def memory_chroma() -> ProviderFixture:
    url = os.getenv("CHROMA_URL")
    if url:
        config = ChromaRemoteImplConfig(url=url)
        provider_type = "remote::chromadb"
    else:
        if not os.getenv("CHROMA_DB_PATH"):
            raise ValueError("CHROMA_DB_PATH or CHROMA_URL must be set")
        config = ChromaInlineImplConfig(db_path=os.getenv("CHROMA_DB_PATH"))
        provider_type = "inline::chromadb"
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="chroma",
                provider_type=provider_type,
                config=config.model_dump(),
            )
        ]
    )


MEMORY_FIXTURES = ["faiss", "pgvector", "weaviate", "remote", "chroma"]


@pytest_asyncio.fixture(scope="session")
async def memory_stack(embedding_model, request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "memory"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    test_stack = await construct_stack_for_test(
        [Api.memory, Api.inference],
        providers,
        provider_data,
        models=[
            ModelInput(
                model_id=embedding_model,
                model_type=ModelType.embedding,
                metadata={
                    "embedding_dimension": get_env_or_fail("EMBEDDING_DIMENSION"),
                },
            )
        ],
    )

    return test_stack.impls[Api.memory], test_stack.impls[Api.memory_banks]
