# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides

from ..inference.fixtures import INFERENCE_FIXTURES
from ..memory.fixtures import MEMORY_FIXTURES
from ..safety.fixtures import SAFETY_FIXTURES
from .fixtures import AGENTS_FIXTURES


DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "meta_reference",
            "safety": "meta_reference",
            "memory": "meta_reference",
            "agents": "meta_reference",
        },
        id="meta_reference",
        marks=pytest.mark.meta_reference,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "safety": "meta_reference",
            "memory": "meta_reference",
            "agents": "meta_reference",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "together",
            "safety": "meta_reference",
            # make this work with Weaviate which is what the together distro supports
            "memory": "meta_reference",
            "agents": "meta_reference",
        },
        id="together",
        marks=pytest.mark.together,
    ),
    pytest.param(
        {
            "inference": "remote",
            "safety": "remote",
            "memory": "remote",
            "agents": "remote",
        },
        id="remote",
        marks=pytest.mark.remote,
    ),
]


def pytest_configure(config):
    for mark in ["meta_reference", "ollama", "together", "remote"]:
        config.addinivalue_line(
            "markers",
            f"{mark}: marks tests as {mark} specific",
        )


def pytest_addoption(parser):
    parser.addoption(
        "--inference-model",
        action="store",
        default="Llama3.1-8B-Instruct",
        help="Specify the inference model to use for testing",
    )
    parser.addoption(
        "--safety-model",
        action="store",
        default="Llama-Guard-3-8B",
        help="Specify the safety model to use for testing",
    )


def pytest_generate_tests(metafunc):
    safety_model = metafunc.config.getoption("--safety-model")
    if "safety_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "safety_model",
            [pytest.param(safety_model, id="")],
            indirect=True,
        )
    if "inference_model" in metafunc.fixturenames:
        inference_model = metafunc.config.getoption("--inference-model")
        models = list(set({inference_model, safety_model}))

        metafunc.parametrize(
            "inference_model",
            [pytest.param(models, id="")],
            indirect=True,
        )
    if "agents_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "safety": SAFETY_FIXTURES,
            "memory": MEMORY_FIXTURES,
            "agents": AGENTS_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("agents_stack", combinations, indirect=True)
