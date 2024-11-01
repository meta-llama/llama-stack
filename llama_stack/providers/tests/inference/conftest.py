# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any, Dict, Tuple

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
from ..env import get_env_or_fail


MODEL_PARAMS = [
    pytest.param("Llama3.1-8B-Instruct", marks=pytest.mark.llama_8b, id="llama_8b"),
    pytest.param("Llama3.2-3B-Instruct", marks=pytest.mark.llama_3b, id="llama_3b"),
]


@pytest.fixture(scope="session", params=MODEL_PARAMS)
def llama_model(request):
    return request.param


@pytest.fixture(scope="session")
def meta_reference(llama_model) -> Provider:
    return Provider(
        provider_id="meta-reference",
        provider_type="meta-reference",
        config=MetaReferenceInferenceConfig(
            model=llama_model,
            max_seq_len=512,
            create_distributed_process_group=False,
            checkpoint_dir=os.getenv("MODEL_CHECKPOINT_DIR", None),
        ).model_dump(),
    )


@pytest.fixture(scope="session")
def ollama(llama_model) -> Provider:
    if llama_model == "Llama3.1-8B-Instruct":
        pytest.skip("Ollama only support Llama3.2-3B-Instruct for testing")

    return Provider(
        provider_id="ollama",
        provider_type="remote::ollama",
        config=(
            OllamaImplConfig(
                host="localhost", port=os.getenv("OLLAMA_PORT", 11434)
            ).model_dump()
        ),
    )


@pytest.fixture(scope="session")
def fireworks(llama_model) -> Provider:
    return Provider(
        provider_id="fireworks",
        provider_type="remote::fireworks",
        config=FireworksImplConfig(
            api_key=get_env_or_fail("FIREWORKS_API_KEY"),
        ).model_dump(),
    )


@pytest.fixture(scope="session")
def together(llama_model) -> Tuple[Provider, Dict[str, Any]]:
    provider = Provider(
        provider_id="together",
        provider_type="remote::together",
        config=TogetherImplConfig().model_dump(),
    )
    return provider, dict(
        together_api_key=get_env_or_fail("TOGETHER_API_KEY"),
    )


PROVIDER_PARAMS = [
    pytest.param("meta_reference", marks=pytest.mark.meta_reference),
    pytest.param("ollama", marks=pytest.mark.ollama),
    pytest.param("fireworks", marks=pytest.mark.fireworks),
    pytest.param("together", marks=pytest.mark.together),
]


@pytest_asyncio.fixture(
    scope="session",
    params=PROVIDER_PARAMS,
)
async def stack_impls(request):
    provider_fixture = request.param
    provider = request.getfixturevalue(provider_fixture)
    if isinstance(provider, tuple):
        provider, provider_data = provider
    else:
        provider_data = None

    impls = await resolve_impls_for_test_v2(
        [Api.inference],
        {"inference": [provider.model_dump()]},
        provider_data,
    )

    return (impls[Api.inference], impls[Api.models])


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llama_8b: mark test to run only with the given model"
    )
    config.addinivalue_line(
        "markers", "llama_3b: mark test to run only with the given model"
    )
    config.addinivalue_line(
        "markers",
        "meta_reference: marks tests as metaref specific",
    )
    config.addinivalue_line(
        "markers",
        "ollama: marks tests as ollama specific",
    )
    config.addinivalue_line(
        "markers",
        "fireworks: marks tests as fireworks specific",
    )
    config.addinivalue_line(
        "markers",
        "together: marks tests as fireworks specific",
    )
