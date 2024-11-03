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
from ..conftest import ProviderFixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def meta_reference() -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="meta-reference",
            provider_type="meta-reference",
            config=FaissImplConfig().model_dump(),
        ),
    )


@pytest.fixture(scope="session")
def pgvector() -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="pgvector",
            provider_type="remote::pgvector",
            config=PGVectorConfig(
                host=os.getenv("PGVECTOR_HOST", "localhost"),
                port=os.getenv("PGVECTOR_PORT", 5432),
                db=get_env_or_fail("PGVECTOR_DB"),
                user=get_env_or_fail("PGVECTOR_USER"),
                password=get_env_or_fail("PGVECTOR_PASSWORD"),
            ).model_dump(),
        ),
    )


@pytest.fixture(scope="session")
def weaviate() -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="weaviate",
            provider_type="remote::weaviate",
            config=WeaviateConfig().model_dump(),
        ),
        provider_data=dict(
            weaviate_api_key=get_env_or_fail("WEAVIATE_API_KEY"),
            weaviate_cluster_url=get_env_or_fail("WEAVIATE_CLUSTER_URL"),
        ),
    )


MEMORY_FIXTURES = ["meta_reference", "pgvector", "weaviate"]

PROVIDER_PARAMS = [
    pytest.param(fixture_name, marks=getattr(pytest.mark, fixture_name))
    for fixture_name in MEMORY_FIXTURES
]


@pytest_asyncio.fixture(
    scope="session",
    params=PROVIDER_PARAMS,
)
async def stack_impls(request):
    fixture_name = request.param
    fixture = request.getfixturevalue(fixture_name)

    impls = await resolve_impls_for_test_v2(
        [Api.memory],
        {"memory": [fixture.provider.model_dump()]},
        fixture.provider_data,
    )

    return impls[Api.memory], impls[Api.memory_banks]


def pytest_configure(config):
    for fixture_name in MEMORY_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )
