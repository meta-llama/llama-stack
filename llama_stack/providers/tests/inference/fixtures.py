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

    return ProviderFixture(
        providers=[
            Provider(
                provider_id=f"meta-reference-{i}",
                provider_type="meta-reference",
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
def inference_ollama(inference_model) -> ProviderFixture:
    inference_model = (
        [inference_model] if isinstance(inference_model, str) else inference_model
    )
    if "Llama3.1-8B-Instruct" in inference_model:
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


INFERENCE_FIXTURES = ["meta_reference", "ollama", "fireworks", "together", "remote"]


@pytest_asyncio.fixture(scope="session")
async def inference_stack(request):
    fixture_name = request.param
    inference_fixture = request.getfixturevalue(f"inference_{fixture_name}")
    impls = await resolve_impls_for_test_v2(
        [Api.inference],
        {"inference": inference_fixture.providers},
        inference_fixture.provider_data,
    )

    return (impls[Api.inference], impls[Api.models])
