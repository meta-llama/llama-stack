# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .fixtures import INFERENCE_FIXTURES


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llama_8b: mark test to run only with the given model"
    )
    config.addinivalue_line(
        "markers", "llama_3b: mark test to run only with the given model"
    )
    for fixture_name in INFERENCE_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )
