# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pytest
import pytest_asyncio

from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import *  # noqa: F403
from llama_stack.apis.eval.eval import ModelCandidate
from llama_stack.distribution.datatypes import *  # noqa: F403

from llama_models.llama3.api import SamplingParams

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
#   pytest -s llama_stack/providers/tests/eval/test_eval.py \
#   --tb=short --disable-warnings
# ```


@pytest_asyncio.fixture(scope="session")
async def eval_settings():
    impls = await resolve_impls_for_test(
        Api.eval, deps=[Api.datasetio, Api.scoring, Api.inference]
    )
    return {
        "eval_impl": impls[Api.eval],
        "scoring_impl": impls[Api.scoring],
        "datasets_impl": impls[Api.datasets],
    }


@pytest.mark.asyncio
async def test_eval(eval_settings):
    datasets_impl = eval_settings["datasets_impl"]
    await register_dataset(
        datasets_impl,
        for_generation=True,
        dataset_id="test_dataset_for_eval",
    )

    response = await datasets_impl.list_datasets()
    assert len(response) == 1

    eval_impl = eval_settings["eval_impl"]
    response = await eval_impl.evaluate_batch(
        dataset_id=response[0].identifier,
        candidate=ModelCandidate(
            model="Llama3.2-1B-Instruct",
            sampling_params=SamplingParams(),
        ),
        scoring_functions=[
            "meta-reference::subset_of",
            "meta-reference::llm_as_judge_8b_correctness",
        ],
    )
    assert response.job_id == "0"
    job_status = await eval_impl.job_status(response.job_id)

    assert job_status and job_status.value == "completed"

    eval_response = await eval_impl.job_result(response.job_id)

    assert eval_response is not None
    assert len(eval_response.generations) == 5
    assert "meta-reference::subset_of" in eval_response.scores
    assert "meta-reference::llm_as_judge_8b_correctness" in eval_response.scores
