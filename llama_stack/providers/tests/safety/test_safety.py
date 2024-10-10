# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.providers.tests.resolver import resolve_impls_for_test

# How to run this test:
#
# 1. Ensure you have a conda with the right dependencies installed. This is a bit tricky
#    since it depends on the provider you are testing. On top of that you need
#    `pytest` and `pytest-asyncio` installed.
#
# 2. Copy and modify the provider_config_example.yaml depending on the provider you are testing.
#
# 3. Run:
#
# ```bash
# PROVIDER_ID=<your_provider> \
#   PROVIDER_CONFIG=provider_config.yaml \
#   pytest -s llama_stack/providers/tests/safety/test_safety.py \
#   --tb=short --disable-warnings
# ```


assert False, "Still WORK IN PROGRESS"


@pytest_asyncio.fixture(scope="session")
async def safety_settings():
    # TODO: make sure we also ask for dependent providers
    impls = await resolve_impls_for_test(
        Api.safety,
    )

    return {
        "impl": impls[Api.safety],
        "shields_impl": impls[Api.shields],
    }


@pytest.fixture
def sample_tool_definition():
    return ToolDefinition(
        tool_name="get_weather",
        description="Get the current weather",
        parameters={
            "location": ToolParamDefinition(
                param_type="string",
                description="The city and state, e.g. San Francisco, CA",
            ),
        },
    )


@pytest.mark.asyncio
async def test_shield_list(safety_settings):
    shields_impl = safety_settings["shields_impl"]
    response = await shields_impl.list_shields()
    assert isinstance(response, list)
    assert len(response) >= 1
    assert all(isinstance(shield, ShieldDefWithProvider) for shield in response)

    model_def = None
    for model in response:
        if model.identifier == params["model"]:
            model_def = model
            break

    assert model_def is not None
    assert model_def.identifier == params["model"]
