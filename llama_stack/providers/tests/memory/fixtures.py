# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile

import pytest
import pytest_asyncio

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.datatypes import Api, Provider, RemoteProviderConfig
from llama_stack.providers.inline.memory.faiss import FaissImplConfig
from llama_stack.providers.remote.memory.pgvector import PGVectorConfig
from llama_stack.providers.remote.memory.weaviate import WeaviateConfig
from llama_stack.providers.tests.resolver import construct_stack_for_test
from llama_stack.providers.utils.kvstore import SqliteKVStoreConfig
from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail
from .fakes import InlineMemoryFakeImpl


@pytest.fixture(scope="session")
def memory_fake() -> ProviderFixture:
    InlineMemoryFakeImpl.stub_method(
        method_name="query_documents",
        return_value_matchers={
            "programming language": QueryDocumentsResponse(
                chunks=[Chunk(content="Python", token_count=1, document_id="")],
                scores=[0.1],
            ),
            "AI and brain-inspired computing": QueryDocumentsResponse(
                chunks=[
                    Chunk(content="neural networks", token_count=2, document_id="")
                ],
                scores=[0.1],
            ),
            "computer": QueryDocumentsResponse(
                chunks=[
                    Chunk(content="test-chunk-1", token_count=1, document_id=""),
                    Chunk(content="test-chunk-2", token_count=1, document_id=""),
                ],
                scores=[0.1, 0.5],
            ),
            "quantum computing": QueryDocumentsResponse(
                chunks=[Chunk(content="Python", token_count=1, document_id="")],
                scores=[0.5],
            ),
        },
    )

    fixture = ProviderFixture(
        providers=[
            Provider(
                provider_id="inline_memory_fake",
                provider_type="test::fake",
                config={},
            )
        ],
    )
    return fixture


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
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="chroma",
                provider_type="remote::chromadb",
                config=RemoteProviderConfig(
                    host=get_env_or_fail("CHROMA_HOST"),
                    port=get_env_or_fail("CHROMA_PORT"),
                ).model_dump(),
            )
        ]
    )


MEMORY_FIXTURES = ["fake", "faiss", "pgvector", "weaviate", "remote", "chroma"]


@pytest_asyncio.fixture(scope="session")
async def memory_stack(request):
    fixture_name = request.param
    fixture = request.getfixturevalue(f"memory_{fixture_name}")

    test_stack = await construct_stack_for_test(
        [Api.memory],
        {"memory": fixture.providers},
        fixture.provider_data,
    )

    return test_stack.impls[Api.memory], test_stack.impls[Api.memory_banks]
