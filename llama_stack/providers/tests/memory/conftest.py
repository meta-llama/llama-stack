# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from .fixtures import MEMORY_FIXTURES


def pytest_configure(config):
    for fixture_name in MEMORY_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


def pytest_generate_tests(metafunc):
    if "memory_stack" in metafunc.fixturenames:
        metafunc.parametrize(
            "memory_stack",
            [
                pytest.param(fixture_name, marks=getattr(pytest.mark, fixture_name))
                for fixture_name in MEMORY_FIXTURES
            ],
            indirect=True,
        )
