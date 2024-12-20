# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides
from ..inference.fixtures import INFERENCE_FIXTURES
from ..memory.fixtures import MEMORY_FIXTURES
from ..safety.fixtures import SAFETY_FIXTURES, safety_model_from_shield
from .fixtures import AGENTS_FIXTURES, TOOL_RUNTIME_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "meta_reference",
            "safety": "llama_guard",
            "memory": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory",
        },
        id="meta_reference",
        marks=pytest.mark.meta_reference,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "safety": "llama_guard",
            "memory": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "together",
            "safety": "llama_guard",
            # make this work with Weaviate which is what the together distro supports
            "memory": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory",
        },
        id="together",
        marks=pytest.mark.together,
    ),
    pytest.param(
        {
            "inference": "fireworks",
            "safety": "llama_guard",
            "memory": "faiss",
            "agents": "meta_reference",
            "tool_runtime": "memory",
        },
        id="fireworks",
        marks=pytest.mark.fireworks,
    ),
    pytest.param(
        {
            "inference": "remote",
            "safety": "remote",
            "memory": "remote",
            "agents": "remote",
            "tool_runtime": "memory",
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


def pytest_addoption(parser):
    parser.addoption(
        "--inference-model",
        action="store",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Specify the inference model to use for testing",
    )
    parser.addoption(
        "--safety-shield",
        action="store",
        default="meta-llama/Llama-Guard-3-1B",
        help="Specify the safety shield to use for testing",
    )


def pytest_generate_tests(metafunc):
    shield_id = metafunc.config.getoption("--safety-shield")
    if "safety_shield" in metafunc.fixturenames:
        metafunc.parametrize(
            "safety_shield",
            [pytest.param(shield_id, id="")],
            indirect=True,
        )
    if "inference_model" in metafunc.fixturenames:
        inference_model = metafunc.config.getoption("--inference-model")
        models = set({inference_model})
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
            "memory": MEMORY_FIXTURES,
            "agents": AGENTS_FIXTURES,
            "tool_runtime": TOOL_RUNTIME_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("agents_stack", combinations, indirect=True)
