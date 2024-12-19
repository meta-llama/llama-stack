# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.apis.models import ModelInput, ModelType
from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.inline.inference.meta_reference import (
    MetaReferenceInferenceConfig,
)
from llama_stack.providers.inline.inference.vllm import VLLMConfig
from llama_stack.providers.remote.inference.bedrock import BedrockConfig

from llama_stack.providers.remote.inference.cerebras import CerebrasImplConfig
from llama_stack.providers.remote.inference.fireworks import FireworksImplConfig
from llama_stack.providers.remote.inference.nvidia import NVIDIAConfig
from llama_stack.providers.remote.inference.ollama import OllamaImplConfig
from llama_stack.providers.remote.inference.tgi import TGIImplConfig
from llama_stack.providers.remote.inference.together import TogetherImplConfig
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.providers.tests.resolver import construct_stack_for_test

from ..conftest import ProviderFixture, remote_stack_fixture
from ..env import get_env_or_fail


@pytest.fixture(scope="session")
def inference_model(request):
    if hasattr(request, "param"):
        return request.param
    return request.config.getoption("--inference-model", None)


@pytest.fixture(scope="session")
def inference_remote() -> ProviderFixture:
    return remote_stack_fixture()


@pytest.fixture(scope="session")
def inference_meta_reference(inference_model) -> ProviderFixture:
    inference_model = (
        [inference_model] if isinstance(inference_model, str) else inference_model
    )
    # If embedding dimension is set, use the 8B model for testing
    if os.getenv("EMBEDDING_DIMENSION"):
        inference_model = ["meta-llama/Llama-3.1-8B-Instruct"]

    return ProviderFixture(
        providers=[
            Provider(
                provider_id=f"meta-reference-{i}",
                provider_type="inline::meta-reference",
                config=MetaReferenceInferenceConfig(
                    model=m,
                    max_seq_len=4096,
                    create_distributed_process_group=False,
                    checkpoint_dir=os.getenv("MODEL_CHECKPOINT_DIR", None),
                ).model_dump(),
            )
            for i, m in enumerate(inference_model)
        ]
    )


@pytest.fixture(scope="session")
def inference_cerebras() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="cerebras",
                provider_type="remote::cerebras",
                config=CerebrasImplConfig(
                    api_key=get_env_or_fail("CEREBRAS_API_KEY"),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_ollama(inference_model) -> ProviderFixture:
    inference_model = (
        [inference_model] if isinstance(inference_model, str) else inference_model
    )
    if inference_model and "Llama3.1-8B-Instruct" in inference_model:
        pytest.skip("Ollama only supports Llama3.2-3B-Instruct for testing")

    return ProviderFixture(
        providers=[
            Provider(
                provider_id="ollama",
                provider_type="remote::ollama",
                config=OllamaImplConfig(
                    host="localhost", port=os.getenv("OLLAMA_PORT", 11434)
                ).model_dump(),
            )
        ],
    )


@pytest_asyncio.fixture(scope="session")
def inference_vllm(inference_model) -> ProviderFixture:
    inference_model = (
        [inference_model] if isinstance(inference_model, str) else inference_model
    )
    return ProviderFixture(
        providers=[
            Provider(
                provider_id=f"vllm-{i}",
                provider_type="inline::vllm",
                config=VLLMConfig(
                    model=m,
                    enforce_eager=True,  # Make test run faster
                ).model_dump(),
            )
            for i, m in enumerate(inference_model)
        ]
    )


@pytest.fixture(scope="session")
def inference_vllm_remote() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="remote::vllm",
                provider_type="remote::vllm",
                config=VLLMInferenceAdapterConfig(
                    url=get_env_or_fail("VLLM_URL"),
                    max_tokens=int(os.getenv("VLLM_MAX_TOKENS", 2048)),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_fireworks() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="fireworks",
                provider_type="remote::fireworks",
                config=FireworksImplConfig(
                    api_key=get_env_or_fail("FIREWORKS_API_KEY"),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_together() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="together",
                provider_type="remote::together",
                config=TogetherImplConfig().model_dump(),
            )
        ],
        provider_data=dict(
            together_api_key=get_env_or_fail("TOGETHER_API_KEY"),
        ),
    )


@pytest.fixture(scope="session")
def inference_bedrock() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="bedrock",
                provider_type="remote::bedrock",
                config=BedrockConfig().model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_nvidia() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="nvidia",
                provider_type="remote::nvidia",
                config=NVIDIAConfig().model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_tgi() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="tgi",
                provider_type="remote::tgi",
                config=TGIImplConfig(
                    url=get_env_or_fail("TGI_URL"),
                    api_token=os.getenv("TGI_API_TOKEN", None),
                ).model_dump(),
            )
        ],
    )


@pytest.fixture(scope="session")
def inference_sentence_transformers() -> ProviderFixture:
    return ProviderFixture(
        providers=[
            Provider(
                provider_id="sentence_transformers",
                provider_type="inline::sentence-transformers",
                config={},
            )
        ]
    )


def get_model_short_name(model_name: str) -> str:
    """Convert model name to a short test identifier.

    Args:
        model_name: Full model name like "Llama3.1-8B-Instruct"

    Returns:
        Short name like "llama_8b" suitable for test markers
    """
    model_name = model_name.lower()
    if "vision" in model_name:
        return "llama_vision"
    elif "3b" in model_name:
        return "llama_3b"
    elif "8b" in model_name:
        return "llama_8b"
    else:
        return model_name.replace(".", "_").replace("-", "_")


@pytest.fixture(scope="session")
def model_id(inference_model) -> str:
    return get_model_short_name(inference_model)


INFERENCE_FIXTURES = [
    "meta_reference",
    "ollama",
    "fireworks",
    "together",
    "vllm",
    "vllm_remote",
    "remote",
    "bedrock",
    "cerebras",
    "nvidia",
    "tgi",
]


@pytest_asyncio.fixture(scope="session")
async def inference_stack(request, inference_model):
    fixture_name = request.param
    inference_fixture = request.getfixturevalue(f"inference_{fixture_name}")
    model_type = ModelType.llm
    metadata = {}
    if os.getenv("EMBEDDING_DIMENSION"):
        model_type = ModelType.embedding
        metadata["embedding_dimension"] = get_env_or_fail("EMBEDDING_DIMENSION")

    test_stack = await construct_stack_for_test(
        [Api.inference],
        {"inference": inference_fixture.providers},
        inference_fixture.provider_data,
        models=[
            ModelInput(
                model_id=inference_model,
                model_type=model_type,
                metadata=metadata,
            )
        ],
    )

    # Pytest yield fixture; see https://docs.pytest.org/en/stable/how-to/fixtures.html#yield-fixtures-recommended
    yield test_stack.impls[Api.inference], test_stack.impls[Api.models]

    # Cleanup code that runs after test case completion
    await test_stack.impls[Api.inference].shutdown()
