# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides

from ..datasetio.fixtures import DATASETIO_FIXTURES

from .fixtures import POST_TRAINING_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "post_training": "torchtune",
            "datasetio": "huggingface",
        },
        id="torchtune_post_training_huggingface_datasetio",
        marks=pytest.mark.torchtune_post_training_huggingface_datasetio,
    ),
]


def pytest_configure(config):
    combined_fixtures = "torchtune_post_training_huggingface_datasetio"
    config.addinivalue_line(
        "markers",
        f"{combined_fixtures}: marks tests as {combined_fixtures} specific",
    )


def pytest_generate_tests(metafunc):
    if "post_training_stack" in metafunc.fixturenames:
        available_fixtures = {
            "eval": POST_TRAINING_FIXTURES,
            "datasetio": DATASETIO_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("post_training_stack", combinations, indirect=True)
