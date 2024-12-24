# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile

import pytest
import pytest_asyncio
from llama_models.llama3.api.datatypes import BuiltinTool

from llama_stack.apis.models import ModelInput, ModelType
from llama_stack.apis.tools import (
    BuiltInToolDef,
    CustomToolDef,
    ToolGroupInput,
    ToolParameter,
    UserDefinedToolGroupDef,
)
from llama_stack.distribution.datatypes import Api, Provider
from llama_stack.providers.inline.agents.meta_reference import (
    MetaReferenceAgentsImplConfig,
)
from llama_stack.providers.tests.resolver import construct_stack_for_test
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig

from ..conftest import ProviderFixture, remote_stack_fixture


def pick_inference_model(inference_model):
    # This is not entirely satisfactory. The fixture `inference_model` can correspond to
    # multiple models when you need to run a safety model in addition to normal agent
    # inference model. We filter off the safety model by looking for "Llama-Guard"
    if isinstance(inference_model, list):
        inference_model = next(m for m in inference_model if "Llama-Guard" not in m)
        assert inference_model is not None
    return inference_model


@pytest.fixture(scope="session")
def agents_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def agents_meta_reference() -> ProviderFixture:
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config=MetaReferenceAgentsImplConfig(
                    # TODO: make this an in-memory store
                    persistence_store=SqliteKVStoreConfig(
                        db_path=sqlite_file.name,
                    ),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def tool_runtime_memory() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="memory-runtime",
                provider_type="inline::memory-runtime",
                config={},
            ),
            Provider(
                provider_id="brave-search",
                provider_type="inline::brave-search",
                config={
                    "api_key": os.environ["BRAVE_SEARCH_API_KEY"],
                },
            ),
            Provider(
                provider_id="tavily-search",
                provider_type="inline::tavily-search",
                config={
                    "api_key": os.environ["TAVILY_SEARCH_API_KEY"],
                },
            ),
            Provider(
                provider_id="code-interpreter",
                provider_type="inline::code-interpreter",
                config={},
            ),
        ],
    )


AGENTS_FIXTURES = ["meta_reference", "remote"]
TOOL_RUNTIME_FIXTURES = ["memory"]


@pytest_asyncio.fixture(scope="session")
async def agents_stack(request, inference_model, safety_shield):
    fixture_dict = request.param

    providers = {}
    provider_data = {}
    for key in ["inference", "safety", "memory", "agents", "tool_runtime"]:
        fixture = request.getfixturevalue(f"{key}_{fixture_dict[key]}")
        providers[key] = fixture.providers
        if key == "inference":
            providers[key].append(
                Provider(
                    provider_id="agents_memory_provider",
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
            provider_id="agents_memory_provider",
            metadata={"embedding_dimension": 384},
        )
    )
    tool_groups = [
        ToolGroupInput(
            tool_group_id="tavily_search_group",
            tool_group=UserDefinedToolGroupDef(
                tools=[
                    BuiltInToolDef(
                        built_in_type=BuiltinTool.brave_search,
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
                        description="memory",
                        parameters=[
                            ToolParameter(
                                name="session_id",
                                description="session id",
                                parameter_type="string",
                                required=True,
                            ),
                            ToolParameter(
                                name="input_messages",
                                description="messages",
                                parameter_type="list",
                                required=True,
                            ),
                            ToolParameter(
                                name="attachments",
                                description="attachments",
                                parameter_type="list",
                                required=False,
                            ),
                        ],
                        metadata={},
                    )
                ],
            ),
            provider_id="memory-runtime",
        ),
        ToolGroupInput(
            tool_group_id="code_interpreter_group",
            tool_group=UserDefinedToolGroupDef(
                tools=[
                    BuiltInToolDef(
                        built_in_type=BuiltinTool.code_interpreter,
                        metadata={},
                    )
                ],
            ),
            provider_id="code-interpreter",
        ),
    ]

    test_stack = await construct_stack_for_test(
        [Api.agents, Api.inference, Api.safety, Api.memory, Api.tool_runtime],
        providers,
        provider_data,
        models=models,
        shields=[safety_shield] if safety_shield else [],
        tool_groups=tool_groups,
    )
    return test_stack
