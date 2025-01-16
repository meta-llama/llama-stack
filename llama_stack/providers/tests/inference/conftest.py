# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import get_provider_fixture_overrides, try_load_config_file_cached
from .fixtures import INFERENCE_FIXTURES


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
    test_config = try_load_config_file_cached(metafunc.config)
    if "inference_model" in metafunc.fixturenames:
        cls_name = metafunc.cls.__name__
        if test_config is not None:
            params = []
            for model in test_config.inference.fixtures.inference_models:
                if ("Vision" in cls_name and "Vision" in model) or (
                    "Vision" not in cls_name and "Vision" not in model
                ):
                    params.append(pytest.param(model, id=model))
        else:
            model = metafunc.config.getoption("--inference-model")
            if model:
                params = [pytest.param(model, id="")]
            else:
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
        if test_config is not None:
            fixtures = [
                (f.get("inference") or f.get("default_fixture_param_id"))
                for f in test_config.inference.fixtures.provider_fixtures
            ]
        elif filtered_stacks := get_provider_fixture_overrides(
            metafunc.config,
            {
                "inference": INFERENCE_FIXTURES,
            },
        ):
            fixtures = [stack.values[0]["inference"] for stack in filtered_stacks]
        else:
            fixtures = INFERENCE_FIXTURES
        metafunc.parametrize("inference_stack", fixtures, indirect=True)
