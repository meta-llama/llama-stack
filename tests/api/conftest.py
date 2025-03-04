# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import copy
import logging
import os
from pathlib import Path

import pytest
from llama_stack_client import LlamaStackClient

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.datatypes import Api
from llama_stack.providers.tests.env import get_env_or_fail

from .fixtures.recordable_mock import RecordableMock
from .report import Report


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True
    # Note:
    # if report_path is not provided (aka no option --report in the pytest command),
    # it will be set to False
    # if --report will give None ( in this case we infer report_path)
    # if --report /a/b is provided, it will be set to the path provided
    # We want to handle all these cases and hence explicitly check for False
    report_path = config.getoption("--report")
    if report_path is not False:
        config.pluginmanager.register(Report(report_path))


TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def pytest_addoption(parser):
    parser.addoption(
        "--report",
        action="store",
        default=False,
        nargs="?",
        type=str,
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )
    parser.addoption(
        "--inference-model",
        default=TEXT_MODEL,
        help="Specify the inference model to use for testing",
    )
    parser.addoption(
        "--vision-inference-model",
        default=VISION_MODEL,
        help="Specify the vision inference model to use for testing",
    )
    parser.addoption(
        "--safety-shield",
        default="meta-llama/Llama-Guard-3-1B",
        help="Specify the safety shield model to use for testing",
    )
    parser.addoption(
        "--embedding-model",
        default=None,
        help="Specify the embedding model to use for testing",
    )
    parser.addoption(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Output dimensionality of the embedding model to use for testing",
    )
    parser.addoption(
        "--record-responses",
        action="store_true",
        default=False,
        help="Record new API responses instead of using cached ones.",
    )


@pytest.fixture(scope="session")
def provider_data():
    keymap = {
        "TAVILY_SEARCH_API_KEY": "tavily_search_api_key",
        "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
        "FIREWORKS_API_KEY": "fireworks_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GROQ_API_KEY": "groq_api_key",
    }
    provider_data = {}
    for key, value in keymap.items():
        if os.environ.get(key):
            provider_data[value] = os.environ[key]
    return provider_data if len(provider_data) > 0 else None


@pytest.fixture(scope="session")
def llama_stack_client(provider_data, text_model_id):
    if os.environ.get("LLAMA_STACK_CONFIG"):
        client = LlamaStackAsLibraryClient(
            get_env_or_fail("LLAMA_STACK_CONFIG"),
            provider_data=provider_data,
            skip_logger_removal=True,
        )
        if not client.initialize():
            raise RuntimeError("Initialization failed")

    elif os.environ.get("LLAMA_STACK_BASE_URL"):
        client = LlamaStackClient(
            base_url=get_env_or_fail("LLAMA_STACK_BASE_URL"),
            provider_data=provider_data,
        )
    else:
        raise ValueError("LLAMA_STACK_CONFIG or LLAMA_STACK_BASE_URL must be set")

    return client


@pytest.fixture(scope="session")
def llama_stack_client_with_mocked_inference(llama_stack_client, request):
    """
    Returns a client with mocked inference APIs and tool runtime APIs that use recorded responses by default.

    If --record-responses is passed, it will call the real APIs and record the responses.
    """
    if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        logging.warning(
            "llama_stack_client_with_mocked_inference is not supported for this client, returning original client without mocking"
        )
        return llama_stack_client

    record_responses = request.config.getoption("--record-responses")
    cache_dir = Path(__file__).parent / "fixtures" / "recorded_responses"

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
def client_with_models(llama_stack_client, text_model_id, vision_model_id, embedding_model_id, embedding_dimension):
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

    if embedding_model_id and embedding_dimension and embedding_model_id not in model_ids:
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
            metadata={"embedding_dimension": embedding_dimension},
        )
    return client


MODEL_SHORT_IDS = {
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "11B",
    "all-MiniLM-L6-v2": "MiniLM",
}


def get_short_id(value):
    return MODEL_SHORT_IDS.get(value, value)


def pytest_generate_tests(metafunc):
    params = []
    values = []
    id_parts = []

    if "text_model_id" in metafunc.fixturenames:
        params.append("text_model_id")
        val = metafunc.config.getoption("--inference-model")
        values.append(val)
        id_parts.append(f"txt={get_short_id(val)}")

    if "vision_model_id" in metafunc.fixturenames:
        params.append("vision_model_id")
        val = metafunc.config.getoption("--vision-inference-model")
        values.append(val)
        id_parts.append(f"vis={get_short_id(val)}")

    if "embedding_model_id" in metafunc.fixturenames:
        params.append("embedding_model_id")
        val = metafunc.config.getoption("--embedding-model")
        values.append(val)
        if val is not None:
            id_parts.append(f"emb={get_short_id(val)}")

    if "embedding_dimension" in metafunc.fixturenames:
        params.append("embedding_dimension")
        val = metafunc.config.getoption("--embedding-dimension")
        values.append(val)
        if val != 384:
            id_parts.append(f"dim={val}")

    if params:
        # Create a single test ID string
        test_id = ":".join(id_parts)
        metafunc.parametrize(params, [values], scope="session", ids=[test_id])
