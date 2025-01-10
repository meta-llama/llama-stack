# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput, ModelType
from llama_stack.apis.tools import ToolGroupInput
from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.tests.resolver import construct_stack_for_test

from ..conftest import ProviderFixture


@pytest.fixture(scope="session")
def tool_runtime_memory_and_search() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="memory-runtime",
                provider_type="inline::memory-runtime",
                config={},
            ),
            Provider(
                provider_id="tavily-search",
                provider_type="remote::tavily-search",
                config={
                    "api_key": os.environ["TAVILY_SEARCH_API_KEY"],
                },
            ),
            Provider(
                provider_id="wolfram-alpha",
                provider_type="remote::wolfram-alpha",
                config={
                    "api_key": os.environ["WOLFRAM_ALPHA_API_KEY"],
                },
            ),
        ],
    )


@pytest.fixture(scope="session")
def tool_group_input_memory() -> ToolGroupInput:
    return ToolGroupInput(
        toolgroup_id="builtin::memory",
        provider_id="memory-runtime",
    )


@pytest.fixture(scope="session")
def tool_group_input_tavily_search() -> ToolGroupInput:
    return ToolGroupInput(
        toolgroup_id="builtin::web_search",
        provider_id="tavily-search",
    )


@pytest.fixture(scope="session")
def tool_group_input_wolfram_alpha() -> ToolGroupInput:
    return ToolGroupInput(
        toolgroup_id="builtin::wolfram_alpha",
        provider_id="wolfram-alpha",
    )


TOOL_RUNTIME_FIXTURES = ["memory_and_search"]


@pytest_asyncio.fixture(scope="session")
async def tools_stack(
    request,
    inference_model,
    tool_group_input_memory,
    tool_group_input_tavily_search,
    tool_group_input_wolfram_alpha,
):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "memory", "tool_runtime"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if key == "inference":
            providers[key].append(
                Provider(
                    provider_id="tools_memory_provider",
                    provider_type="inline::sentence-transformers",
                    config={},
                )
            )
        if fixture.provider_data:
            provider_data.update(fixture.provider_data)
    inference_models = (
        inference_model if isinstance(inference_model, list) else [inference_model]
    )
    models = [
        ModelInput(
            model_id=model,
            model_type=ModelType.llm,
            provider_id=providers["inference"][0].provider_id,
        )
        for model in inference_models
    ]
    models.append(
        ModelInput(
            model_id="all-MiniLM-L6-v2",
            model_type=ModelType.embedding,
            provider_id="tools_memory_provider",
            metadata={"embedding_dimension": 384},
        )
    )

    test_stack = await construct_stack_for_test(
        [Api.tool_groups, Api.inference, Api.memory, Api.tool_runtime],
        providers,
        provider_data,
        models=models,
        tool_groups=[
            tool_group_input_tavily_search,
            tool_group_input_wolfram_alpha,
            tool_group_input_memory,
        ],
    )
    return test_stack
