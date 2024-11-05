# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.adapters.memory.pgvector import PGVectorConfig
from llama_stack.providers.adapters.memory.weaviate import WeaviateConfig
from llama_stack.providers.impls.meta_reference.memory import FaissImplConfig

from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2
from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def memory_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def memory_meta_reference() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="meta-reference",
                config=FaissImplConfig().model_dump(),
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


MEMORY_FIXTURES = ["meta_reference", "pgvector", "weaviate", "remote"]


@pytest_asyncio.fixture(scope="session")
async def memory_stack(request):
    fixture_name = request.param
    fixture = request.getfixturevalue(f"memory_{fixture_name}")

    impls = await resolve_impls_for_test_v2(
        [Api.memory],
        {"memory": fixture.providers},
        fixture.provider_data,
    )

    return impls[Api.memory], impls[Api.memory_banks]
