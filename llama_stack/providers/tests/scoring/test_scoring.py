# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pytest
import pytest_asyncio

from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403

from llama_stack.providers.tests.datasetio.test_datasetio import register_dataset
from llama_stack.providers.tests.resolver import resolve_impls_for_test

# How to run this test:
#
# 1. Ensure you have a conda with the right dependencies installed. This is a bit tricky
#    since it depends on the provider you are testing. On top of that you need
#    `pytest` and `pytest-asyncio` installed.
#
# 2. Copy and modify the provider_config_example.yaml depending on the provider you are testing.
#
# 3. Run:
#
# ```bash
# PROVIDER_ID=<your_provider> \
#   PROVIDER_CONFIG=provider_config.yaml \
#   pytest -s llama_stack/providers/tests/scoring/test_scoring.py \
#   --tb=short --disable-warnings
# ```


@pytest_asyncio.fixture(scope="session")
async def scoring_settings():
    impls = await resolve_impls_for_test(Api.scoring, deps=[Api.datasetio])
    return {
        "scoring_impl": impls[Api.scoring],
        "scoring_functions_impl": impls[Api.scoring_functions],
        "datasets_impl": impls[Api.datasets],
    }


@pytest.mark.asyncio
async def test_scoring_functions_list(scoring_settings):
    scoring_functions_impl = scoring_settings["scoring_functions_impl"]
    scoring_functions = await scoring_functions_impl.list_scoring_functions()
    assert isinstance(scoring_functions, list)
    assert len(scoring_functions) > 0
    function_ids = [f.identifier for f in scoring_functions]
    assert "equality" in function_ids


@pytest.mark.asyncio
async def test_scoring_score(scoring_settings):
    scoring_impl = scoring_settings["scoring_impl"]
    datasets_impl = scoring_settings["datasets_impl"]
    await register_dataset(datasets_impl)

    response = await datasets_impl.list_datasets()
    assert len(response) == 1

    response = await scoring_impl.score_batch(
        dataset_id=response[0].identifier,
        scoring_functions=["equality"],
    )

    assert len(response.results) == 1
    assert "equality" in response.results
