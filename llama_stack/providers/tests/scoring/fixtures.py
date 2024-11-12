# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2
from ..conftest import ProviderFixture, remote_stack_fixture


@pytest.fixture(scope="session")
def scoring_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def scoring_meta_reference() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="meta-reference",
                config={},
            )
        ],
    )


@pytest.fixture(scope="session")
def scoring_braintrust() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="braintrust",
                provider_type="braintrust",
                config={},
            )
        ],
    )


SCORING_FIXTURES = ["meta_reference", "remote", "braintrust"]


@pytest_asyncio.fixture(scope="session")
async def scoring_stack(request, inference_model):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["datasetio", "scoring", "inference"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)

    impls = await resolve_impls_for_test_v2(
        [Api.scoring, Api.datasetio, Api.inference],
        providers,
        provider_data,
    )

    provider_id = providers["inference"][0].provider_id
    print(f"Registering model {inference_model} with provider {provider_id}")
    await impls[Api.models].register_model(
        model_id=inference_model,
        provider_id=provider_id,
    )
    await impls[Api.models].register_model(
        model_id="Llama3.1-405B-Instruct",
        provider_id=provider_id,
    )
    await impls[Api.models].register_model(
        model_id="Llama3.1-8B-Instruct",
        provider_id=provider_id,
    )

    return impls
