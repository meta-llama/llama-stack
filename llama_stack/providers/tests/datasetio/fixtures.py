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
def datasetio_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def datasetio_localfs() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="localfs",
                provider_type="inline::localfs",
                config={},
            )
        ],
    )


@pytest.fixture(scope="session")
def datasetio_huggingface() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="huggingface",
                provider_type="remote::huggingface",
                config={},
            )
        ],
    )


DATASETIO_FIXTURES = ["localfs", "remote", "huggingface"]


@pytest_asyncio.fixture(scope="session")
async def datasetio_stack(request):
    fixture_name = request.param
    fixture = request.getfixturevalue(f"datasetio_{fixture_name}")

    test_stack = await construct_stack_for_test(
        [Api.datasetio],
        {"datasetio": fixture.providers},
        fixture.provider_data,
    )

    return test_stack.impls[Api.datasetio], test_stack.impls[Api.datasets]
