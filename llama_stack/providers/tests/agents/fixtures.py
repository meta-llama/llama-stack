# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.impls.meta_reference.agents import (
    MetaReferenceAgentsImplConfig,
)

from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

from ..conftest import ProviderFixture, remote_stack_fixture


@pytest.fixture(scope="session")
def agents_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def agents_meta_reference() -> ProviderFixture:
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="meta-reference",
                config=MetaReferenceAgentsImplConfig(
                    # TODO: make this an in-memory store
                    persistence_store=SqliteKVStoreConfig(
                        db_path=sqlite_file.name,
                    ),
                ).model_dump(),
            )
        ],
    )


AGENTS_FIXTURES = ["meta_reference", "remote"]


@pytest_asyncio.fixture(scope="session")
async def agents_stack(request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "safety", "memory", "agents"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    impls = await resolve_impls_for_test_v2(
        [Api.agents, Api.inference, Api.safety, Api.memory],
        providers,
        provider_data,
    )
    return impls[Api.agents], impls[Api.memory]
