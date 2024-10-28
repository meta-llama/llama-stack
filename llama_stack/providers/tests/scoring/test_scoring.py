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
    impls = await resolve_impls_for_test(
        Api.scoring, deps=[Api.datasetio, Api.inference]
    )
    return {
        "scoring_impl": impls[Api.scoring],
        "scoring_functions_impl": impls[Api.scoring_functions],
        "datasets_impl": impls[Api.datasets],
    }


@pytest_asyncio.fixture(scope="session")
async def provider_scoring_functions():
    return {
        "meta-reference": {
            "meta-reference::equality",
            "meta-reference::subset_of",
            "meta-reference::llm_as_judge_8b_correctness",
        },
        "braintrust": {
            "braintrust::factuality",
            "braintrust::answer-correctness",
        },
    }


@pytest.mark.asyncio
async def test_scoring_functions_list(scoring_settings, provider_scoring_functions):
    scoring_impl = scoring_settings["scoring_impl"]
    scoring_functions_impl = scoring_settings["scoring_functions_impl"]
    scoring_functions = await scoring_functions_impl.list_scoring_functions()
    assert isinstance(scoring_functions, list)
    assert len(scoring_functions) > 0
    function_ids = [f.identifier for f in scoring_functions]
    # get current provider_type we're testing
    provider = scoring_impl.routing_table.get_provider_impl(function_ids[0])
    provider_type = provider.__provider_spec__.provider_type

    for x in provider_scoring_functions[provider_type]:
        assert x in function_ids


@pytest.mark.asyncio
async def test_scoring_functions_register(scoring_settings):
    scoring_impl = scoring_settings["scoring_impl"]
    scoring_functions_impl = scoring_settings["scoring_functions_impl"]
    datasets_impl = scoring_settings["datasets_impl"]

    # get current provider_type we're testing
    scoring_functions = await scoring_functions_impl.list_scoring_functions()
    function_ids = [f.identifier for f in scoring_functions]
    provider = scoring_impl.routing_table.get_provider_impl(function_ids[0])
    provider_type = provider.__provider_spec__.provider_type
    if provider_type not in ("meta-reference"):
        pytest.skip(
            "Other scoring providers don't support registering scoring functions."
        )

    test_prompt = """Output a number between 0 to 10. Your answer must match the format \n Number: <answer>"""
    # register the scoring function
    await scoring_functions_impl.register_scoring_function(
        ScoringFnDefWithProvider(
            identifier="meta-reference::llm_as_judge_8b_random",
            description="Llm As Judge Scoring Function",
            parameters=[],
            return_type=NumberType(),
            context=LLMAsJudgeContext(
                prompt_template=test_prompt,
                judge_model="Llama3.1-8B-Instruct",
                judge_score_regex=[r"Number: (\d+)"],
            ),
            provider_id="test-meta",
        )
    )

    scoring_functions = await scoring_functions_impl.list_scoring_functions()
    assert isinstance(scoring_functions, list)
    assert len(scoring_functions) > 0
    function_ids = [f.identifier for f in scoring_functions]
    assert "meta-reference::llm_as_judge_8b_random" in function_ids

    # test score using newly registered scoring function
    await register_dataset(datasets_impl)
    response = await datasets_impl.list_datasets()
    assert len(response) == 1
    response = await scoring_impl.score_batch(
        dataset_id=response[0].identifier,
        scoring_functions=[
            "meta-reference::llm_as_judge_8b_random",
        ],
    )
    assert "meta-reference::llm_as_judge_8b_random" in response.results


@pytest.mark.asyncio
async def test_scoring_score(scoring_settings, provider_scoring_functions):
    scoring_impl = scoring_settings["scoring_impl"]
    datasets_impl = scoring_settings["datasets_impl"]
    scoring_functions_impl = scoring_settings["scoring_functions_impl"]
    await register_dataset(datasets_impl)

    response = await datasets_impl.list_datasets()
    assert len(response) == 1

    # get current provider_type we're testing
    scoring_functions = await scoring_functions_impl.list_scoring_functions()
    function_ids = [f.identifier for f in scoring_functions]
    provider = scoring_impl.routing_table.get_provider_impl(function_ids[0])
    provider_type = provider.__provider_spec__.provider_type

    response = await scoring_impl.score_batch(
        dataset_id=response[0].identifier,
        scoring_functions=list(provider_scoring_functions[provider_type]),
    )

    assert len(response.results) == len(provider_scoring_functions[provider_type])
    for x in provider_scoring_functions[provider_type]:
        assert x in response.results
