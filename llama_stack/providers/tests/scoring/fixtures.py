# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.scoring.braintrust import BraintrustScoringConfig
from llama_stack.providers.tests.resolver import construct_stack_for_test
from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def scoring_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def scoring_basic() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="basic",
                provider_type="inline::basic",
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
                provider_type="inline::braintrust",
                config=BraintrustScoringConfig(
                    openai_api_key=get_env_or_fail("OPENAI_API_KEY"),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def scoring_llm_as_judge() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="llm-as-judge",
                provider_type="inline::llm-as-judge",
                config={},
            )
        ],
    )


SCORING_FIXTURES = ["basic", "remote", "braintrust", "llm_as_judge"]


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

    test_stack = await construct_stack_for_test(
        [Api.scoring, Api.datasetio, Api.inference],
        providers,
        provider_data,
        models=[
            ModelInput(model_id=model)
            for model in [
                inference_model,
                "Llama3.1-405B-Instruct",
                "Llama3.1-8B-Instruct",
            ]
        ],
    )

    return test_stack.impls
