# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.distribution.datatypes import Api, Provider

from llama_stack.providers.adapters.inference.fireworks import FireworksImplConfig
from llama_stack.providers.adapters.inference.ollama import OllamaImplConfig
from llama_stack.providers.adapters.inference.together import TogetherImplConfig
from llama_stack.providers.impls.meta_reference.inference import (
    MetaReferenceInferenceConfig,
)
from llama_stack.providers.tests.resolver import resolve_impls_for_test_v2
from ..conftest import ProviderFixture
from ..env import get_env_or_fail


MODEL_PARAMS = [
    pytest.param("Llama3.1-8B-Instruct", marks=pytest.mark.llama_8b, id="llama_8b"),
    pytest.param("Llama3.2-3B-Instruct", marks=pytest.mark.llama_3b, id="llama_3b"),
]


@pytest.fixture(scope="session", params=MODEL_PARAMS)
def inference_model(request):
    return request.param


@pytest.fixture(scope="session")
def inference_meta_reference(inference_model) -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="meta-reference",
            provider_type="meta-reference",
            config=MetaReferenceInferenceConfig(
                model=inference_model,
                max_seq_len=512,
                create_distributed_process_group=False,
                checkpoint_dir=os.getenv("MODEL_CHECKPOINT_DIR", None),
            ).model_dump(),
        ),
    )


@pytest.fixture(scope="session")
def inference_ollama(inference_model) -> ProviderFixture:
    if inference_model == "Llama3.1-8B-Instruct":
        pytest.skip("Ollama only support Llama3.2-3B-Instruct for testing")

    return ProviderFixture(
        provider=Provider(
            provider_id="ollama",
            provider_type="remote::ollama",
            config=OllamaImplConfig(
                host="localhost", port=os.getenv("OLLAMA_PORT", 11434)
            ).model_dump(),
        ),
    )


@pytest.fixture(scope="session")
def inference_fireworks(inference_model) -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="fireworks",
            provider_type="remote::fireworks",
            config=FireworksImplConfig(
                api_key=get_env_or_fail("FIREWORKS_API_KEY"),
            ).model_dump(),
        ),
    )


@pytest.fixture(scope="session")
def inference_together(inference_model) -> ProviderFixture:
    return ProviderFixture(
        provider=Provider(
            provider_id="together",
            provider_type="remote::together",
            config=TogetherImplConfig().model_dump(),
        ),
        provider_data=dict(
            together_api_key=get_env_or_fail("TOGETHER_API_KEY"),
        ),
    )


INFERENCE_FIXTURES = ["meta_reference", "ollama", "fireworks", "together"]


@pytest_asyncio.fixture(scope="session", params=INFERENCE_FIXTURES)
async def inference_stack(request):
    fixture_name = request.param
    inference_fixture = request.getfixturevalue(f"inference_{fixture_name}")
    impls = await resolve_impls_for_test_v2(
        [Api.inference],
        {"inference": [inference_fixture.provider.model_dump()]},
        inference_fixture.provider_data,
    )

    return (impls[Api.inference], impls[Api.models])
