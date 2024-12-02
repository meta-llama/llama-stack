# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.tests.resolver import construct_stack_for_test
from ..conftest import ProviderFixture, remote_stack_fixture


@pytest.fixture(scope="session")
def eval_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def eval_meta_reference() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config={},
            )
        ],
    )


EVAL_FIXTURES = ["meta_reference", "remote"]


@pytest_asyncio.fixture(scope="session")
async def eval_stack(request):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in [
        "datasetio",
        "eval",
        "scoring",
        "inference",
        "agents",
        "safety",
        "memory",
    ]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    test_stack = await construct_stack_for_test(
        [
            Api.eval,
            Api.datasetio,
            Api.inference,
            Api.scoring,
            Api.agents,
            Api.safety,
            Api.memory,
        ],
        providers,
        provider_data,
    )

    return test_stack.impls
