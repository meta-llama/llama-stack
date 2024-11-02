# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any, Dict, Tuple

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.adapters.memory.pgvector import PGVectorConfig
from llama_stack.providers.adapters.memory.weaviate import WeaviateConfig
from llama_stack.providers.impls.meta_reference.memory import FaissImplConfig

from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def meta_reference() -> Provider:
    return Provider(
        provider_id="meta-reference",
        provider_type="meta-reference",
        config=FaissImplConfig().model_dump(),
    )


@pytest.fixture(scope="session")
def pgvector() -> Provider:
    return Provider(
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


@pytest.fixture(scope="session")
def weaviate() -> Tuple[Provider, Dict[str, Any]]:
    provider = Provider(
        provider_id="weaviate",
        provider_type="remote::weaviate",
        config=WeaviateConfig().model_dump(),
    )
    return provider, dict(
        weaviate_api_key=get_env_or_fail("WEAVIATE_API_KEY"),
        weaviate_cluster_url=get_env_or_fail("WEAVIATE_CLUSTER_URL"),
    )


PROVIDER_PARAMS = [
    pytest.param("meta_reference", marks=pytest.mark.meta_reference),
    pytest.param("pgvector", marks=pytest.mark.pgvector),
    pytest.param("weaviate", marks=pytest.mark.weaviate),
]


@pytest_asyncio.fixture(
    scope="session",
    params=PROVIDER_PARAMS,
)
async def stack_impls(request):
    provider_fixture = request.param
    provider = request.getfixturevalue(provider_fixture)
    if isinstance(provider, tuple):
        provider, provider_data = provider
    else:
        provider_data = None

    impls = await resolve_impls_for_test_v2(
        [Api.memory],
        {"memory": [provider.model_dump()]},
        provider_data,
    )

    return impls[Api.memory], impls[Api.memory_banks]


def pytest_configure(config):
    config.addinivalue_line("markers", "pgvector: marks tests as pgvector specific")
    config.addinivalue_line(
        "markers",
        "meta_reference: marks tests as metaref specific",
    )
    config.addinivalue_line(
        "markers",
        "weaviate: marks tests as weaviate specific",
    )
