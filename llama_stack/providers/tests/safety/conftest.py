# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides

from ..inference.fixtures import INFERENCE_FIXTURES
from .fixtures import SAFETY_FIXTURES


DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "meta_reference",
            "safety": "meta_reference",
        },
        id="meta_reference",
        marks=pytest.mark.meta_reference,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "safety": "meta_reference",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "together",
            "safety": "together",
        },
        id="together",
        marks=pytest.mark.together,
    ),
]


def pytest_configure(config):
    for mark in ["meta_reference", "ollama", "together"]:
        config.addinivalue_line(
            "markers",
            f"{mark}: marks tests as {mark} specific",
        )


def pytest_generate_tests(metafunc):
    # We use this method to make sure we have built-in simple combos for safety tests
    # But a user can also pass in a custom combination via the CLI by doing
    #  `--providers inference=together,safety=meta_reference`

    if "safety_stack" in metafunc.fixturenames:
        # print(f"metafunc.fixturenames: {metafunc.fixturenames}, {metafunc}")
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "safety": SAFETY_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("safety_stack", combinations, indirect=True)
