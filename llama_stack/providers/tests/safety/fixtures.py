# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.shields import Shield, ShieldType

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.safety.meta_reference import (
    LlamaGuardShieldConfig,
    SafetyConfig,
)
from llama_stack.providers.remote.safety.bedrock import BedrockSafetyConfig
from llama_stack.providers.tests.env import get_env_or_fail
from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2

from ..conftest import ProviderFixture, remote_stack_fixture


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
def safety_bedrock() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="bedrock",
                provider_type="remote::bedrock",
                config=BedrockSafetyConfig().model_dump(),
            )
        ],
    )


SAFETY_FIXTURES = ["meta_reference", "bedrock", "remote"]


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

    safety_impl = impls[Api.safety]
    shields_impl = impls[Api.shields]

    # Register the appropriate shield based on provider type
    provider_id = safety_fixture.providers[0].provider_id
    provider_type = safety_fixture.providers[0].provider_type

    shield_config = {}
    identifier = "llama_guard"
    if provider_type == "meta-reference":
        shield_config["model"] = safety_model
    elif provider_type == "remote::together":
        shield_config["model"] = safety_model
    elif provider_type == "remote::bedrock":
        identifier = get_env_or_fail("BEDROCK_GUARDRAIL_IDENTIFIER")
        shield_config["guardrailVersion"] = get_env_or_fail("BEDROCK_GUARDRAIL_VERSION")

    # Create shield
    shield = Shield(
        identifier=identifier,
        shield_type=ShieldType.llama_guard,
        provider_id=provider_id,
        params=shield_config,
    )

    return safety_impl, shields_impl, shield
