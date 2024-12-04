# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides

from .fixtures import INFERENCE_FIXTURES


def pytest_addoption(parser):
    parser.addoption(
        "--inference-model",
        action="store",
        default=None,
        help="Specify the inference model to use for testing",
    )


def pytest_configure(config):
    for model in ["llama_8b", "llama_3b", "llama_vision"]:
        config.addinivalue_line(
            "markers", f"{model}: mark test to run only with the given model"
        )

    for fixture_name in INFERENCE_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


MODEL_PARAMS = [
    pytest.param(
        "meta-llama/Llama-3.1-8B-Instruct", marks=pytest.mark.llama_8b, id="llama_8b"
    ),
    pytest.param(
        "meta-llama/Llama-3.2-3B-Instruct", marks=pytest.mark.llama_3b, id="llama_3b"
    ),
]

VISION_MODEL_PARAMS = [
    pytest.param(
        "Llama3.2-11B-Vision-Instruct",
        marks=pytest.mark.llama_vision,
        id="llama_vision",
    ),
]


def pytest_generate_tests(metafunc):
    if "inference_model" in metafunc.fixturenames:
        model = metafunc.config.getoption("--inference-model")
        if model:
            params = [pytest.param(model, id="")]
        else:
            cls_name = metafunc.cls.__name__
            if "Vision" in cls_name:
                params = VISION_MODEL_PARAMS
            else:
                params = MODEL_PARAMS

        metafunc.parametrize(
            "inference_model",
            params,
            indirect=True,
        )
    if "inference_stack" in metafunc.fixturenames:
        fixtures = INFERENCE_FIXTURES
        if filtered_stacks := get_provider_fixture_overrides(
            metafunc.config,
            {
                "inference": INFERENCE_FIXTURES,
            },
        ):
            fixtures = [stack.values[0]["inference"] for stack in filtered_stacks]
        metafunc.parametrize("inference_stack", fixtures, indirect=True)
