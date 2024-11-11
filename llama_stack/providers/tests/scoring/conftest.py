# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides

from ..datasetio.fixtures import DATASETIO_FIXTURES
from ..inference.fixtures import INFERENCE_FIXTURES
from .fixtures import SCORING_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "scoring": "meta_reference",
            "datasetio": "localfs",
            "inference": "fireworks",
        },
        id="meta_reference_scoring_fireworks_inference",
        marks=pytest.mark.meta_reference_scoring_fireworks_inference,
    ),
    pytest.param(
        {
            "scoring": "meta_reference",
            "datasetio": "localfs",
            "inference": "together",
        },
        id="meta_reference_scoring_together_inference",
        marks=pytest.mark.meta_reference_scoring_together_inference,
    ),
    pytest.param(
        {
            "scoring": "braintrust",
            "datasetio": "localfs",
            "inference": "together",
        },
        id="braintrust_scoring_together_inference",
        marks=pytest.mark.braintrust_scoring_together_inference,
    ),
]


def pytest_configure(config):
    for fixture_name in [
        "meta_reference_scoring_fireworks_inference",
        "meta_reference_scoring_together_inference",
        "braintrust_scoring_together_inference",
    ]:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


def pytest_addoption(parser):
    parser.addoption(
        "--inference-model",
        action="store",
        default="Llama3.2-3B-Instruct",
        help="Specify the inference model to use for testing",
    )


def pytest_generate_tests(metafunc):
    if "scoring_stack" in metafunc.fixturenames:
        available_fixtures = {
            "scoring": SCORING_FIXTURES,
            "datasetio": DATASETIO_FIXTURES,
            "inference": INFERENCE_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("scoring_stack", combinations, indirect=True)
