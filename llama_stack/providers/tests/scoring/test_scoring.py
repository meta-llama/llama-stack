# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.scoring_functions import (
    AggregationFunctionType,
    BasicScoringFnParams,
    LLMAsJudgeScoringFnParams,
    RegexParserScoringFnParams,
)
from llama_stack.distribution.datatypes import Api
from llama_stack.providers.tests.datasetio.test_datasetio import register_dataset

# How to run this test:
#
# pytest llama_stack/providers/tests/scoring/test_scoring.py
#   -m "meta_reference"
#   -v -s --tb=short --disable-warnings


@pytest.fixture
def sample_judge_prompt_template():
    return "Output a number response in the following format: Score: <number>, where <number> is the number between 0 and 9."


class TestScoring:
    @pytest.mark.asyncio
    async def test_scoring_functions_list(self, scoring_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        scoring_functions_impl = scoring_stack[Api.scoring_functions]
        response = await scoring_functions_impl.list_scoring_functions()
        assert isinstance(response, list)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_scoring_score(self, scoring_stack):
        (
            scoring_impl,
            scoring_functions_impl,
            datasetio_impl,
            datasets_impl,
            models_impl,
        ) = (
            scoring_stack[Api.scoring],
            scoring_stack[Api.scoring_functions],
            scoring_stack[Api.datasetio],
            scoring_stack[Api.datasets],
            scoring_stack[Api.models],
        )
        scoring_fns_list = await scoring_functions_impl.list_scoring_functions()
        provider_id = scoring_fns_list[0].provider_id
        if provider_id == "llm-as-judge":
            pytest.skip(
                f"{provider_id} provider does not support scoring without params"
            )

        await register_dataset(datasets_impl, for_rag=True)
        response = await datasets_impl.list_datasets()
        assert len(response) == 1

        # scoring individual rows
        rows = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=3,
        )
        assert len(rows.rows) == 3

        scoring_fns_list = await scoring_functions_impl.list_scoring_functions()
        scoring_functions = {
            scoring_fns_list[0].identifier: None,
        }

        response = await scoring_impl.score(
            input_rows=rows.rows,
            scoring_functions=scoring_functions,
        )
        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == len(rows.rows)

        # score batch
        response = await scoring_impl.score_batch(
            dataset_id="test_dataset",
            scoring_functions=scoring_functions,
        )
        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == 5

    @pytest.mark.asyncio
    async def test_scoring_score_with_params_llm_as_judge(
        self, scoring_stack, sample_judge_prompt_template, judge_model
    ):
        (
            scoring_impl,
            scoring_functions_impl,
            datasetio_impl,
            datasets_impl,
            models_impl,
        ) = (
            scoring_stack[Api.scoring],
            scoring_stack[Api.scoring_functions],
            scoring_stack[Api.datasetio],
            scoring_stack[Api.datasets],
            scoring_stack[Api.models],
        )
        await register_dataset(datasets_impl, for_rag=True)
        response = await datasets_impl.list_datasets()
        assert len(response) == 1

        scoring_fns_list = await scoring_functions_impl.list_scoring_functions()
        provider_id = scoring_fns_list[0].provider_id
        if provider_id == "braintrust" or provider_id == "basic":
            pytest.skip(f"{provider_id} provider does not support scoring with params")

        # scoring individual rows
        rows = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=3,
        )
        assert len(rows.rows) == 3

        scoring_functions = {
            "llm-as-judge::base": LLMAsJudgeScoringFnParams(
                judge_model=judge_model,
                prompt_template=sample_judge_prompt_template,
                judge_score_regexes=[r"Score: (\d+)"],
                aggregation_functions=[AggregationFunctionType.categorical_count],
            )
        }

        response = await scoring_impl.score(
            input_rows=rows.rows,
            scoring_functions=scoring_functions,
        )
        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == len(rows.rows)

        # score batch
        response = await scoring_impl.score_batch(
            dataset_id="test_dataset",
            scoring_functions=scoring_functions,
        )
        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == 5

    @pytest.mark.asyncio
    async def test_scoring_score_with_aggregation_functions(
        self, scoring_stack, sample_judge_prompt_template, judge_model
    ):
        (
            scoring_impl,
            scoring_functions_impl,
            datasetio_impl,
            datasets_impl,
            models_impl,
        ) = (
            scoring_stack[Api.scoring],
            scoring_stack[Api.scoring_functions],
            scoring_stack[Api.datasetio],
            scoring_stack[Api.datasets],
            scoring_stack[Api.models],
        )
        await register_dataset(datasets_impl, for_rag=True)
        rows = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=3,
        )
        assert len(rows.rows) == 3

        scoring_fns_list = await scoring_functions_impl.list_scoring_functions()
        scoring_functions = {}
        aggr_fns = [
            AggregationFunctionType.accuracy,
            AggregationFunctionType.median,
            AggregationFunctionType.categorical_count,
            AggregationFunctionType.average,
        ]
        for x in scoring_fns_list:
            if x.provider_id == "llm-as-judge":
                aggr_fns = [AggregationFunctionType.categorical_count]
                scoring_functions[x.identifier] = LLMAsJudgeScoringFnParams(
                    judge_model=judge_model,
                    prompt_template=sample_judge_prompt_template,
                    judge_score_regexes=[r"Score: (\d+)"],
                    aggregation_functions=aggr_fns,
                )
            elif x.provider_id == "basic" or x.provider_id == "braintrust":
                if "regex_parser" in x.identifier:
                    scoring_functions[x.identifier] = RegexParserScoringFnParams(
                        aggregation_functions=aggr_fns,
                    )
                else:
                    scoring_functions[x.identifier] = BasicScoringFnParams(
                        aggregation_functions=aggr_fns,
                    )
            else:
                scoring_functions[x.identifier] = None

        response = await scoring_impl.score(
            input_rows=rows.rows,
            scoring_functions=scoring_functions,
        )

        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == len(rows.rows)
            assert len(response.results[x].aggregated_results) == len(aggr_fns)
