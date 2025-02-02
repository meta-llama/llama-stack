# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides, get_test_config_for_api
from .fixtures import INFERENCE_FIXTURES


def pytest_configure(config):
    for model in ["llama_8b", "llama_3b", "llama_vision"]:
        config.addinivalue_line("markers", f"{model}: mark test to run only with the given model")

    for fixture_name in INFERENCE_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


MODEL_PARAMS = [
    pytest.param("meta-llama/Llama-3.1-8B-Instruct", marks=pytest.mark.llama_8b, id="llama_8b"),
    pytest.param("meta-llama/Llama-3.2-3B-Instruct", marks=pytest.mark.llama_3b, id="llama_3b"),
]

VISION_MODEL_PARAMS = [
    pytest.param(
        "Llama3.2-11B-Vision-Instruct",
        marks=pytest.mark.llama_vision,
        id="llama_vision",
    ),
]


def pytest_generate_tests(metafunc):
    test_config = get_test_config_for_api(metafunc.config, "inference")

    if "inference_model" in metafunc.fixturenames:
        cls_name = metafunc.cls.__name__
        params = []
        inference_models = getattr(test_config, "inference_models", [])
        for model in inference_models:
            if ("Vision" in cls_name and "Vision" in model) or ("Vision" not in cls_name and "Vision" not in model):
                params.append(pytest.param(model, id=model))

        if not params:
            model = metafunc.config.getoption("--inference-model")
            params = [pytest.param(model, id="")]

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
        if test_config:
            if custom_fixtures := [
                (scenario.fixture_combo_id or scenario.provider_fixtures.get("inference"))
                for scenario in test_config.scenarios
            ]:
                fixtures = custom_fixtures
        metafunc.parametrize("inference_stack", fixtures, indirect=True)
