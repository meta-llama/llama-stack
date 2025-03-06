# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from ..datasetio.test_datasetio import register_dataset


@pytest.fixture
def sample_judge_prompt_template():
    return "Output a number response in the following format: Score: <number>, where <number> is the number between 0 and 9."


@pytest.fixture
def sample_scoring_fn_id():
    return "llm-as-judge-test-prompt"


def register_scoring_function(
    llama_stack_client,
    provider_id,
    scoring_fn_id,
    judge_model_id,
    judge_prompt_template,
):
    llama_stack_client.scoring_functions.register(
        scoring_fn_id=scoring_fn_id,
        provider_id=provider_id,
        description="LLM as judge scoring function with test prompt",
        return_type={
            "type": "string",
        },
        params={
            "type": "llm_as_judge",
            "judge_model": judge_model_id,
            "prompt_template": judge_prompt_template,
        },
    )


def test_scoring_functions_list(llama_stack_client):
    response = llama_stack_client.scoring_functions.list()
    assert isinstance(response, list)
    assert len(response) > 0


def test_scoring_functions_register(
    llama_stack_client,
    sample_scoring_fn_id,
    judge_model_id,
    sample_judge_prompt_template,
):
    llm_as_judge_provider = [
        x
        for x in llama_stack_client.providers.list()
        if x.api == "scoring" and x.provider_type == "inline::llm-as-judge"
    ]
    if len(llm_as_judge_provider) == 0:
        pytest.skip("No llm-as-judge provider found, cannot test registeration")

    llm_as_judge_provider_id = llm_as_judge_provider[0].provider_id
    register_scoring_function(
        llama_stack_client,
        llm_as_judge_provider_id,
        sample_scoring_fn_id,
        judge_model_id,
        sample_judge_prompt_template,
    )

    list_response = llama_stack_client.scoring_functions.list()
    assert isinstance(list_response, list)
    assert len(list_response) > 0
    assert any(x.identifier == sample_scoring_fn_id for x in list_response)

    # TODO: add unregister to make clean state


def test_scoring_score(llama_stack_client):
    register_dataset(llama_stack_client, for_rag=True)
    response = llama_stack_client.datasets.list()
    assert len(response) == 1

    # scoring individual rows
    rows = llama_stack_client.datasetio.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=3,
    )
    assert len(rows.rows) == 3

    scoring_fns_list = llama_stack_client.scoring_functions.list()
    scoring_functions = {
        scoring_fns_list[0].identifier: None,
    }

    response = llama_stack_client.scoring.score(
        input_rows=rows.rows,
        scoring_functions=scoring_functions,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows.rows)

    # score batch
    response = llama_stack_client.scoring.score_batch(
        dataset_id="test_dataset",
        scoring_functions=scoring_functions,
        save_results_dataset=False,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == 5


def test_scoring_score_with_params_llm_as_judge(llama_stack_client, sample_judge_prompt_template, judge_model_id):
    register_dataset(llama_stack_client, for_rag=True)
    response = llama_stack_client.datasets.list()
    assert len(response) == 1

    # scoring individual rows
    rows = llama_stack_client.datasetio.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=3,
    )
    assert len(rows.rows) == 3

    scoring_functions = {
        "llm-as-judge::base": dict(
            type="llm_as_judge",
            judge_model=judge_model_id,
            prompt_template=sample_judge_prompt_template,
            judge_score_regexes=[r"Score: (\d+)"],
            aggregation_functions=[
                "categorical_count",
            ],
        )
    }

    response = llama_stack_client.scoring.score(
        input_rows=rows.rows,
        scoring_functions=scoring_functions,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows.rows)

    # score batch
    response = llama_stack_client.scoring.score_batch(
        dataset_id="test_dataset",
        scoring_functions=scoring_functions,
        save_results_dataset=False,
    )
    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == 5


@pytest.mark.skip(reason="Skipping because this seems to be really slow")
def test_scoring_score_with_aggregation_functions(llama_stack_client, sample_judge_prompt_template, judge_model_id):
    register_dataset(llama_stack_client, for_rag=True)
    rows = llama_stack_client.datasetio.get_rows_paginated(
        dataset_id="test_dataset",
        rows_in_page=3,
    )
    assert len(rows.rows) == 3

    scoring_fns_list = llama_stack_client.scoring_functions.list()
    scoring_functions = {}
    aggr_fns = [
        "accuracy",
        "median",
        "categorical_count",
        "average",
    ]
    for x in scoring_fns_list:
        if x.provider_id == "llm-as-judge":
            aggr_fns = ["categorical_count"]
            scoring_functions[x.identifier] = dict(
                type="llm_as_judge",
                judge_model=judge_model_id,
                prompt_template=sample_judge_prompt_template,
                judge_score_regexes=[r"Score: (\d+)"],
                aggregation_functions=aggr_fns,
            )
        elif x.provider_id == "basic" or x.provider_id == "braintrust":
            if "regex_parser" in x.identifier:
                scoring_functions[x.identifier] = dict(
                    type="regex_parser",
                    parsing_regexes=[r"Score: (\d+)"],
                    aggregation_functions=aggr_fns,
                )
            else:
                scoring_functions[x.identifier] = dict(
                    type="basic",
                    aggregation_functions=aggr_fns,
                )
        else:
            scoring_functions[x.identifier] = None

    response = llama_stack_client.scoring.score(
        input_rows=rows.rows,
        scoring_functions=scoring_functions,
    )

    assert len(response.results) == len(scoring_functions)
    for x in scoring_functions:
        assert x in response.results
        assert len(response.results[x].score_rows) == len(rows.rows)
        assert len(response.results[x].aggregated_results) == len(aggr_fns)
