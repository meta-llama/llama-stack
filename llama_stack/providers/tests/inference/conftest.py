# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from .fixtures import INFERENCE_FIXTURES


def pytest_addoption(parser):
    parser.addoption(
        "--inference-model",
        action="store",
        default=None,
        help="Specify the inference model to use for testing",
    )


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


MODEL_PARAMS = [
    pytest.param("Llama3.1-8B-Instruct", marks=pytest.mark.llama_8b, id="llama_8b"),
    pytest.param("Llama3.2-3B-Instruct", marks=pytest.mark.llama_3b, id="llama_3b"),
]


def pytest_generate_tests(metafunc):
    if "inference_model" in metafunc.fixturenames:
        model = metafunc.config.getoption("--inference-model")
        if model:
            params = [pytest.param(model, id="")]
        else:
            params = MODEL_PARAMS

        metafunc.parametrize(
            "inference_model",
            params,
            indirect=True,
        )
    if "inference_stack" in metafunc.fixturenames:
        metafunc.parametrize(
            "inference_stack",
            [
                pytest.param(fixture_name, marks=getattr(pytest.mark, fixture_name))
                for fixture_name in INFERENCE_FIXTURES
            ],
            indirect=True,
        )
