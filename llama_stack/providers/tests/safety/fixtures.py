# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.adapters.safety.together import TogetherSafetyConfig
from llama_stack.providers.impls.meta_reference.safety import (
    LlamaGuardShieldConfig,
    SafetyConfig,
)

from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2

from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def safety_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def safety_model(request):
    if hasattr(request, "param"):
        return request.param
    return request.config.getoption("--safety-model", None)


@pytest.fixture(scope="session")
def safety_meta_reference(safety_model) -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="meta-reference",
                config=SafetyConfig(
                    llama_guard_shield=LlamaGuardShieldConfig(
                        model=safety_model,
                    ),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def safety_together() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="together",
                provider_type="remote::together",
                config=TogetherSafetyConfig().model_dump(),
            )
        ],
        provider_data=dict(
            together_api_key=get_env_or_fail("TOGETHER_API_KEY"),
        ),
    )


SAFETY_FIXTURES = ["meta_reference", "together", "remote"]


@pytest_asyncio.fixture(scope="session")
async def safety_stack(inference_model, safety_model, request):
    # We need an inference + safety fixture to test safety
    fixture_dict = request.param
    inference_fixture = request.getfixturevalue(
        f"inference_{fixture_dict['inference']}"
    )
    safety_fixture = request.getfixturevalue(f"safety_{fixture_dict['safety']}")

    providers = {
        "inference": inference_fixture.providers,
        "safety": safety_fixture.providers,
    }
    provider_data = {}
    if inference_fixture.provider_data:
        provider_data.update(inference_fixture.provider_data)
    if safety_fixture.provider_data:
        provider_data.update(safety_fixture.provider_data)

    impls = await resolve_impls_for_test_v2(
        [Api.safety, Api.shields, Api.inference],
        providers,
        provider_data,
    )
    return impls[Api.safety], impls[Api.shields]
