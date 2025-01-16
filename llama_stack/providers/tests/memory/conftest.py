# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import (
    get_provider_fixture_overrides,
    get_provider_fixtures_from_config,
    try_load_config_file_cached,
)

from ..inference.fixtures import INFERENCE_FIXTURES
from .fixtures import MEMORY_FIXTURES


DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "sentence_transformers",
            "memory": "faiss",
        },
        id="sentence_transformers",
        marks=pytest.mark.sentence_transformers,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "memory": "faiss",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "sentence_transformers",
            "memory": "chroma",
        },
        id="chroma",
        marks=pytest.mark.chroma,
    ),
    pytest.param(
        {
            "inference": "bedrock",
            "memory": "qdrant",
        },
        id="qdrant",
        marks=pytest.mark.qdrant,
    ),
    pytest.param(
        {
            "inference": "fireworks",
            "memory": "weaviate",
        },
        id="weaviate",
        marks=pytest.mark.weaviate,
    ),
]


def pytest_configure(config):
    for fixture_name in MEMORY_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


def pytest_generate_tests(metafunc):
    test_config = try_load_config_file_cached(metafunc.config)
    provider_fixtures_config = (
        test_config.memory.fixtures.provider_fixtures
        if test_config is not None and test_config.memory is not None
        else None
    )
    custom_fixtures = (
        get_provider_fixtures_from_config(
            provider_fixtures_config, DEFAULT_PROVIDER_COMBINATIONS
        )
        if provider_fixtures_config is not None
        else None
    )
    if "embedding_model" in metafunc.fixturenames:
        model = metafunc.config.getoption("--embedding-model")
        if model:
            params = [pytest.param(model, id="")]
        else:
            params = [pytest.param("all-MiniLM-L6-v2", id="")]

        metafunc.parametrize("embedding_model", params, indirect=True)

    if "memory_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "memory": MEMORY_FIXTURES,
        }
        combinations = (
            custom_fixtures
            or get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("memory_stack", combinations, indirect=True)
