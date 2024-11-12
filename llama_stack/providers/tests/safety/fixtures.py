# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput

from llama_stack.apis.shields import ShieldInput, ShieldType

from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.safety.llama_guard import LlamaGuardConfig
from llama_stack.providers.inline.safety.prompt_guard import PromptGuardConfig
from llama_stack.providers.remote.safety.bedrock import BedrockSafetyConfig

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
def safety_llama_guard(safety_model) -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="inline::llama-guard",
                provider_type="inline::llama-guard",
                config=LlamaGuardConfig(model=safety_model).model_dump(),
            )
        ],
    )


# TODO: this is not tested yet; we would need to configure the run_shield() test
# and parametrize it with the "prompt" for testing depending on the safety fixture
# we are using.
@pytest.fixture(scope="session")
def safety_prompt_guard() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="inline::prompt-guard",
                provider_type="inline::prompt-guard",
                config=PromptGuardConfig().model_dump(),
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


SAFETY_FIXTURES = ["llama_guard", "bedrock", "remote"]


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

    shield_provider_type = safety_fixture.providers[0].provider_type
    shield_input = get_shield_to_register(shield_provider_type, safety_model)

    impls = await resolve_impls_for_test_v2(
        [Api.safety, Api.shields, Api.inference],
        providers,
        provider_data,
        models=[ModelInput(model_id=inference_model)],
        shields=[shield_input],
    )

    shield = await impls[Api.shields].get_shield(shield_input.shield_id)
    return impls[Api.safety], impls[Api.shields], shield


def get_shield_to_register(provider_type: str, safety_model: str) -> ShieldInput:
    shield_config = {}
    shield_type = ShieldType.llama_guard
    identifier = "llama_guard"
    if provider_type == "meta-reference":
        shield_config["model"] = safety_model
    elif provider_type == "remote::together":
        shield_config["model"] = safety_model
    elif provider_type == "remote::bedrock":
        identifier = get_env_or_fail("BEDROCK_GUARDRAIL_IDENTIFIER")
        shield_config["guardrailVersion"] = get_env_or_fail("BEDROCK_GUARDRAIL_VERSION")
        shield_type = ShieldType.generic_content_shield

    return ShieldInput(
        shield_id=identifier,
        shield_type=shield_type,
        params=shield_config,
    )
