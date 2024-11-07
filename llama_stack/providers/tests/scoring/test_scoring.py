# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.providers.tests.datasetio.test_datasetio import register_dataset

# How to run this test:
#
# pytest llama_stack/providers/tests/scoring/test_scoring.py
#   -m "meta_reference"
#   -v -s --tb=short --disable-warnings


class TestScoring:
    @pytest.mark.asyncio
    async def test_scoring_functions_list(self, scoring_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        _, scoring_functions_impl, _, _ = scoring_stack
        response = await scoring_functions_impl.list_scoring_functions()
        assert isinstance(response, list)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_scoring_score(self, scoring_stack):
        scoring_impl, scoring_functions_impl, datasetio_impl, datasets_impl = (
            scoring_stack
        )
        await register_dataset(datasets_impl)
        response = await datasets_impl.list_datasets()
        assert len(response) == 1

        # scoring individual rows
        rows = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset",
            rows_in_page=3,
        )
        assert len(rows.rows) == 3

        scoring_functions = [
            "meta-reference::llm_as_judge_8b_correctness",
            "meta-reference::equality",
        ]
        response = await scoring_impl.score(
            input_rows=rows.rows,
            scoring_functions=scoring_functions,
        )
        assert len(response.results) == len(scoring_functions)
        for x in scoring_functions:
            assert x in response.results
            assert len(response.results[x].score_rows) == len(rows.rows)
