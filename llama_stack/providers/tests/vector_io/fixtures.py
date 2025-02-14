# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput, ModelType
from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.vector_io.chroma import ChromaVectorIOConfig as InlineChromaVectorIOConfig
from llama_stack.providers.inline.vector_io.faiss import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec import SQLiteVectorIOConfig
from llama_stack.providers.remote.vector_io.chroma import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector import PGVectorVectorIOConfig
from llama_stack.providers.remote.vector_io.qdrant import QdrantVectorIOConfig
from llama_stack.providers.remote.vector_io.weaviate import WeaviateVectorIOConfig
from llama_stack.providers.tests.resolver import construct_stack_for_test
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def embedding_model(request):
    if hasattr(request, "param"):
        return request.param
    return request.config.getoption("--embedding-model", None)


@pytest.fixture(scope="session")
def vector_io_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def vector_io_faiss() -> ProviderFixture:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="faiss",
                provider_type="inline::faiss",
                config=FaissVectorIOConfig(
                    kvstore=SqliteKVStoreConfig(db_path=temp_file.name).model_dump(),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def vector_io_sqlite_vec() -> ProviderFixture:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="sqlite_vec",
                provider_type="inline::sqlite_vec",
                config=SQLiteVectorIOConfig(
                    kvstore=SqliteKVStoreConfig(db_path=temp_file.name).model_dump(),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def vector_io_pgvector() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="pgvector",
                provider_type="remote::pgvector",
                config=PGVectorVectorIOConfig(
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
def vector_io_weaviate() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="weaviate",
                provider_type="remote::weaviate",
                config=WeaviateVectorIOConfig().model_dump(),
            )
        ],
        provider_data=dict(
            weaviate_api_key=get_env_or_fail("WEAVIATE_API_KEY"),
            weaviate_cluster_url=get_env_or_fail("WEAVIATE_CLUSTER_URL"),
        ),
    )


@pytest.fixture(scope="session")
def vector_io_chroma() -> ProviderFixture:
    url = os.getenv("CHROMA_URL")
    if url:
        config = ChromaVectorIOConfig(url=url)
        provider_type = "remote::chromadb"
    else:
        if not os.getenv("CHROMA_DB_PATH"):
            raise ValueError("CHROMA_DB_PATH or CHROMA_URL must be set")
        config = InlineChromaVectorIOConfig(db_path=os.getenv("CHROMA_DB_PATH"))
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


@pytest.fixture(scope="session")
def vector_io_qdrant() -> ProviderFixture:
    url = os.getenv("QDRANT_URL")
    if url:
        config = QdrantVectorIOConfig(url=url)
        provider_type = "remote::qdrant"
    else:
        raise ValueError("QDRANT_URL must be set")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="qdrant",
                provider_type=provider_type,
                config=config.model_dump(),
            )
        ]
    )


VECTOR_IO_FIXTURES = ["faiss", "pgvector", "weaviate", "chroma", "qdrant", "sqlite_vec"]


@pytest_asyncio.fixture(scope="session")
async def vector_io_stack(embedding_model, request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "vector_io"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    test_stack = await construct_stack_for_test(
        [Api.vector_io, Api.inference],
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

    return test_stack.impls[Api.vector_io], test_stack.impls[Api.vector_dbs]
