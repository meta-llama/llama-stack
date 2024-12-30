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
from .fixtures import TOOL_RUNTIME_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "together",
            "safety": "llama_guard",
            "memory": "faiss",
            "tool_runtime": "memory_and_search",
        },
        id="together",
        marks=pytest.mark.together,
    ),
]


def pytest_configure(config):
    for mark in ["together"]:
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
    if "tools_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "safety": SAFETY_FIXTURES,
            "memory": MEMORY_FIXTURES,
            "tool_runtime": TOOL_RUNTIME_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        print(combinations)
        metafunc.parametrize("tools_stack", combinations, indirect=True)
