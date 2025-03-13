# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
import inspect
import logging
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from llama_stack_client import LlamaStackClient

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.datatypes import Api
from llama_stack.distribution.stack import run_config_from_adhoc_config_spec
from llama_stack.env import get_env_or_fail

from .recordable_mock import RecordableMock


@pytest.fixture(scope="session")
def provider_data():
    # TODO: this needs to be generalized so each provider can have a sample provider data just
    # like sample run config on which we can do replace_env_vars()
    keymap = {
        "TAVILY_SEARCH_API_KEY": "tavily_search_api_key",
        "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
        "FIREWORKS_API_KEY": "fireworks_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GROQ_API_KEY": "groq_api_key",
        "WOLFRAM_ALPHA_API_KEY": "wolfram_alpha_api_key",
    }
    provider_data = {}
    for key, value in keymap.items():
        if os.environ.get(key):
            provider_data[value] = os.environ[key]
    return provider_data


@pytest.fixture(scope="session")
def llama_stack_client_with_mocked_inference(llama_stack_client, request):
    """
    Returns a client with mocked inference APIs and tool runtime APIs that use recorded responses by default.

    If --record-responses is passed, it will call the real APIs and record the responses.
    """
    # TODO: will rework this to be more stable
    return llama_stack_client
    if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        logging.warning(
            "llama_stack_client_with_mocked_inference is not supported for this client, returning original client without mocking"
        )
        return llama_stack_client

    record_responses = request.config.getoption("--record-responses")
    cache_dir = Path(__file__).parent / "recorded_responses"

    # Create a shallow copy of the client to avoid modifying the original
    client = copy.copy(llama_stack_client)

    # Get the inference API used by the agents implementation
    agents_impl = client.async_client.impls[Api.agents]
    original_inference = agents_impl.inference_api

    # Create a new inference object with the same attributes
    inference_mock = copy.copy(original_inference)

    # Replace the methods with recordable mocks
    inference_mock.chat_completion = RecordableMock(
        original_inference.chat_completion, cache_dir, "chat_completion", record=record_responses
    )
    inference_mock.completion = RecordableMock(
        original_inference.completion, cache_dir, "text_completion", record=record_responses
    )
    inference_mock.embeddings = RecordableMock(
        original_inference.embeddings, cache_dir, "embeddings", record=record_responses
    )

    # Replace the inference API in the agents implementation
    agents_impl.inference_api = inference_mock

    original_tool_runtime_api = agents_impl.tool_runtime_api
    tool_runtime_mock = copy.copy(original_tool_runtime_api)

    # Replace the methods with recordable mocks
    tool_runtime_mock.invoke_tool = RecordableMock(
        original_tool_runtime_api.invoke_tool, cache_dir, "invoke_tool", record=record_responses
    )
    agents_impl.tool_runtime_api = tool_runtime_mock

    # Also update the client.inference for consistency
    client.inference = inference_mock

    return client


@pytest.fixture(scope="session")
def inference_provider_type(llama_stack_client):
    providers = llama_stack_client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    assert len(inference_providers) > 0, "No inference providers found"
    return inference_providers[0].provider_type


@pytest.fixture(scope="session")
def client_with_models(
    llama_stack_client,
    text_model_id,
    vision_model_id,
    embedding_model_id,
    embedding_dimension,
    judge_model_id,
):
    client = llama_stack_client

    providers = [p for p in client.providers.list() if p.api == "inference"]
    assert len(providers) > 0, "No inference providers found"
    inference_providers = [p.provider_id for p in providers if p.provider_type != "inline::sentence-transformers"]

    model_ids = {m.identifier for m in client.models.list()}
    model_ids.update(m.provider_resource_id for m in client.models.list())

    if text_model_id and text_model_id not in model_ids:
        client.models.register(model_id=text_model_id, provider_id=inference_providers[0])
    if vision_model_id and vision_model_id not in model_ids:
        client.models.register(model_id=vision_model_id, provider_id=inference_providers[0])
    if judge_model_id and judge_model_id not in model_ids:
        client.models.register(model_id=judge_model_id, provider_id=inference_providers[0])

    if embedding_model_id and embedding_model_id not in model_ids:
        # try to find a provider that supports embeddings, if sentence-transformers is not available
        selected_provider = None
        for p in providers:
            if p.provider_type == "inline::sentence-transformers":
                selected_provider = p
                break

        selected_provider = selected_provider or providers[0]
        client.models.register(
            model_id=embedding_model_id,
            provider_id=selected_provider.provider_id,
            model_type="embedding",
            metadata={"embedding_dimension": embedding_dimension or 384},
        )
    return client


@pytest.fixture(scope="session")
def available_shields(llama_stack_client):
    return [shield.identifier for shield in llama_stack_client.shields.list()]


@pytest.fixture(scope="session")
def model_providers(llama_stack_client):
    return {x.provider_id for x in llama_stack_client.providers.list() if x.api == "inference"}


@pytest.fixture(autouse=True)
def skip_if_no_model(request):
    model_fixtures = ["text_model_id", "vision_model_id", "embedding_model_id", "judge_model_id"]
    test_func = request.node.function

    actual_params = inspect.signature(test_func).parameters.keys()
    for fixture in model_fixtures:
        # Only check fixtures that are actually in the test function's signature
        if fixture in actual_params and fixture in request.fixturenames and not request.getfixturevalue(fixture):
            pytest.skip(f"{fixture} empty - skipping test")


@pytest.fixture(scope="session")
def llama_stack_client(request, provider_data, text_model_id):
    config = request.config.getoption("--stack-config")
    if not config:
        config = get_env_or_fail("LLAMA_STACK_CONFIG")

    if not config:
        raise ValueError("You must specify either --stack-config or LLAMA_STACK_CONFIG")

    # check if this looks like a URL
    if config.startswith("http") or "//" in config:
        return LlamaStackClient(
            base_url=config,
            provider_data=provider_data,
        )

    if "=" in config:
        run_config = run_config_from_adhoc_config_spec(config)
        run_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(run_config_file.name, "w") as f:
            yaml.dump(run_config.model_dump(), f)
        config = run_config_file.name

    client = LlamaStackAsLibraryClient(
        config,
        provider_data=provider_data,
        skip_logger_removal=True,
    )
    if not client.initialize():
        raise RuntimeError("Initialization failed")

    return client
