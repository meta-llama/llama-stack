# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides
from ..inference.fixtures import INFERENCE_FIXTURES
from ..safety.fixtures import SAFETY_FIXTURES
from ..vector_io.fixtures import VECTOR_IO_FIXTURES
from .fixtures import TOOL_RUNTIME_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "together",
            "safety": "llama_guard",
            "vector_io": "faiss",
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


def pytest_generate_tests(metafunc):
    if "tools_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "safety": SAFETY_FIXTURES,
            "vector_io": VECTOR_IO_FIXTURES,
            "tool_runtime": TOOL_RUNTIME_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures) or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("tools_stack", combinations, indirect=True)
