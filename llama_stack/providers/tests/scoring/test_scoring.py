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
    }


@pytest.mark.asyncio
async def test_scoring_functions_list(scoring_settings):
    # NOTE: this needs you to ensure that you are starting from a clean state
    # but so far we don't have an unregister API unfortunately, so be careful
    scoring_functions_impl = scoring_settings["scoring_functions_impl"]
    response = await scoring_functions_impl.list_scoring_functions()
    assert isinstance(response, list)
    assert len(response) == 0
