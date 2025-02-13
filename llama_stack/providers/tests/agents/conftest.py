# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import (
    get_provider_fixture_overrides,
    get_provider_fixture_overrides_from_test_config,
    get_test_config_for_api,
)
from ..inference.fixtures import INFERENCE_FIXTURES
from ..safety.fixtures import SAFETY_FIXTURES, safety_model_from_shield
from ..tools.fixtures import TOOL_RUNTIME_FIXTURES
from ..vector_io.fixtures import VECTOR_IO_FIXTURES
from .fixtures import AGENTS_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "meta_reference",
            "safety": "llama_guard",
            "vector_io": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory_and_search",
        },
        id="meta_reference",
        marks=pytest.mark.meta_reference,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "safety": "llama_guard",
            "vector_io": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory_and_search",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "together",
            "safety": "llama_guard",
            # make this work with Weaviate which is what the together distro supports
            "vector_io": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory_and_search",
        },
        id="together",
        marks=pytest.mark.together,
    ),
    pytest.param(
        {
            "inference": "fireworks",
            "safety": "llama_guard",
            "vector_io": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory_and_search",
        },
        id="fireworks",
        marks=pytest.mark.fireworks,
    ),
    pytest.param(
        {
            "inference": "remote",
            "safety": "remote",
            "vector_io": "remote",
            "agents": "remote",
            "tool_runtime": "memory_and_search",
        },
        id="remote",
        marks=pytest.mark.remote,
    ),
]


def pytest_configure(config):
    for mark in ["meta_reference", "ollama", "together", "fireworks", "remote"]:
        config.addinivalue_line(
            "markers",
            f"{mark}: marks tests as {mark} specific",
        )


def pytest_generate_tests(metafunc):
    test_config = get_test_config_for_api(metafunc.config, "agents")
    shield_id = getattr(test_config, "safety_shield", None) or metafunc.config.getoption("--safety-shield")
    inference_models = getattr(test_config, "inference_models", None) or [
        metafunc.config.getoption("--inference-model")
    ]

    if "safety_shield" in metafunc.fixturenames:
        metafunc.parametrize(
            "safety_shield",
            [pytest.param(shield_id, id="")],
            indirect=True,
        )
    if "inference_model" in metafunc.fixturenames:
        models = set(inference_models)
        if safety_model := safety_model_from_shield(shield_id):
            models.add(safety_model)

        metafunc.parametrize(
            "inference_model",
            [pytest.param(list(models), id="")],
            indirect=True,
        )
    if "agents_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "safety": SAFETY_FIXTURES,
            "vector_io": VECTOR_IO_FIXTURES,
            "agents": AGENTS_FIXTURES,
            "tool_runtime": TOOL_RUNTIME_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides_from_test_config(metafunc.config, "agents", DEFAULT_PROVIDER_COMBINATIONS)
            or get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("agents_stack", combinations, indirect=True)
