# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput, ModelType
from llama_stack.apis.tools import (
    BuiltInToolDef,
    CustomToolDef,
    ToolGroupInput,
    ToolParameter,
    UserDefinedToolGroupDef,
)
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
        ],
    )


TOOL_RUNTIME_FIXTURES = ["memory_and_search"]


@pytest_asyncio.fixture(scope="session")
async def tools_stack(request, inference_model, safety_shield):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "memory", "tools", "tool_runtime"]:
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

    tool_groups = [
        ToolGroupInput(
            tool_group_id="tavily_search_group",
            tool_group=UserDefinedToolGroupDef(
                tools=[
                    BuiltInToolDef(
                        name="brave_search",
                        description="Search the web using Brave Search",
                        metadata={},
                    ),
                ],
            ),
            provider_id="tavily-search",
        ),
        ToolGroupInput(
            tool_group_id="memory_group",
            tool_group=UserDefinedToolGroupDef(
                tools=[
                    CustomToolDef(
                        name="memory",
                        description="Query the memory bank",
                        parameters=[
                            ToolParameter(
                                name="query",
                                description="The query to search for in memory",
                                parameter_type="string",
                                required=True,
                            ),
                            ToolParameter(
                                name="memory_bank_id",
                                description="The ID of the memory bank to search",
                                parameter_type="string",
                                required=True,
                            ),
                        ],
                        metadata={},
                    )
                ],
            ),
            provider_id="memory-runtime",
        ),
    ]

    test_stack = await construct_stack_for_test(
        [Api.tools, Api.inference, Api.memory],
        providers,
        provider_data,
        models=models,
        tool_groups=tool_groups,
    )
    return test_stack
