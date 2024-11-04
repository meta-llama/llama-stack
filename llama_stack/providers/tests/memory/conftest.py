# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .fixtures import MEMORY_FIXTURES


def pytest_configure(config):
    for fixture_name in MEMORY_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )
