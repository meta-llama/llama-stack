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

from ..conftest import ProviderFixture


@pytest.fixture(scope="session")
def agents_meta_reference(inference_model, safety_model) -> ProviderFixture:
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        provider=Provider(
            provider_id="meta-reference",
            provider_type="meta-reference",
            config=MetaReferenceAgentsImplConfig(
                # TODO: make this an in-memory store
                persistence_store=SqliteKVStoreConfig(
                    db_path=sqlite_file.name,
                ),
            ).model_dump(),
        ),
    )


AGENTS_FIXTURES = ["meta_reference"]


@pytest_asyncio.fixture(scope="session")
async def agents_stack(inference_model, safety_model, request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["agents", "inference", "safety", "memory"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = [fixture.provider]
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    impls = await resolve_impls_for_test_v2(
        [Api.agents, Api.inference, Api.safety, Api.memory],
        providers,
        provider_data,
    )
    return impls[Api.agents], impls[Api.memory]
