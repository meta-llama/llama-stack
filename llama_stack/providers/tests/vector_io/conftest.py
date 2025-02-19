# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ..conftest import (
    get_provider_fixture_overrides,
    get_provider_fixture_overrides_from_test_config,
    get_test_config_for_api,
)
from ..inference.fixtures import INFERENCE_FIXTURES
from .fixtures import VECTOR_IO_FIXTURES

DEFAULT_PROVIDER_COMBINATIONS = [
    pytest.param(
        {
            "inference": "sentence_transformers",
            "vector_io": "faiss",
        },
        id="sentence_transformers",
        marks=pytest.mark.sentence_transformers,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "vector_io": "pgvector",
        },
        id="pgvector",
        marks=pytest.mark.pgvector,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "vector_io": "faiss",
        },
        id="ollama",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "vector_io": "sqlite_vec",
        },
        id="sqlite_vec",
        marks=pytest.mark.ollama,
    ),
    pytest.param(
        {
            "inference": "sentence_transformers",
            "vector_io": "chroma",
        },
        id="chroma",
        marks=pytest.mark.chroma,
    ),
    pytest.param(
        {
            "inference": "ollama",
            "vector_io": "qdrant",
        },
        id="qdrant",
        marks=pytest.mark.qdrant,
    ),
    pytest.param(
        {
            "inference": "fireworks",
            "vector_io": "weaviate",
        },
        id="weaviate",
        marks=pytest.mark.weaviate,
    ),
]


def pytest_configure(config):
    for fixture_name in VECTOR_IO_FIXTURES:
        config.addinivalue_line(
            "markers",
            f"{fixture_name}: marks tests as {fixture_name} specific",
        )


def pytest_generate_tests(metafunc):
    test_config = get_test_config_for_api(metafunc.config, "vector_io")
    if "embedding_model" in metafunc.fixturenames:
        model = getattr(test_config, "embedding_model", None)
        # Fall back to the default if not specified by the config file
        model = model or metafunc.config.getoption("--embedding-model")
        if model:
            params = [pytest.param(model, id="")]
        else:
            params = [pytest.param("all-minilm:l6-v2", id="")]

        metafunc.parametrize("embedding_model", params, indirect=True)

    if "vector_io_stack" in metafunc.fixturenames:
        available_fixtures = {
            "inference": INFERENCE_FIXTURES,
            "vector_io": VECTOR_IO_FIXTURES,
        }
        combinations = (
            get_provider_fixture_overrides_from_test_config(metafunc.config, "vector_io", DEFAULT_PROVIDER_COMBINATIONS)
            or get_provider_fixture_overrides(metafunc.config, available_fixtures)
            or DEFAULT_PROVIDER_COMBINATIONS
        )
        metafunc.parametrize("vector_io_stack", combinations, indirect=True)
