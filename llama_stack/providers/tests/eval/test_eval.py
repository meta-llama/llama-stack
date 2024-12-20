# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.type_system import ChatCompletionInputType, StringType

from llama_stack.apis.eval.eval import (
    AppEvalTaskConfig,
    BenchmarkEvalTaskConfig,
    ModelCandidate,
)
from llama_stack.apis.inference import SamplingParams
from llama_stack.apis.scoring_functions import LLMAsJudgeScoringFnParams
from llama_stack.distribution.datatypes import Api
from llama_stack.providers.tests.datasetio.test_datasetio import register_dataset
from .constants import JUDGE_PROMPT

# How to run this test:
#
# pytest llama_stack/providers/tests/eval/test_eval.py
#   -m "meta_reference_eval_together_inference_huggingface_datasetio"
#   -v -s --tb=short --disable-warnings


class Testeval:
    @pytest.mark.asyncio
    async def test_eval_tasks_list(self, eval_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        eval_tasks_impl = eval_stack[Api.eval_tasks]
        response = await eval_tasks_impl.list_eval_tasks()
        assert isinstance(response, list)

    @pytest.mark.asyncio
    async def test_eval_evaluate_rows(self, eval_stack, inference_model, judge_model):
        eval_impl, eval_tasks_impl, datasetio_impl, datasets_impl, models_impl = (
            eval_stack[Api.eval],
            eval_stack[Api.eval_tasks],
            eval_stack[Api.datasetio],
            eval_stack[Api.datasets],
            eval_stack[Api.models],
        )

        await register_dataset(
            datasets_impl, for_generation=True, dataset_id="test_dataset_for_eval"
        )
        response = await datasets_impl.list_datasets()

        rows = await datasetio_impl.get_rows_paginated(
            dataset_id="test_dataset_for_eval",
            rows_in_page=3,
        )
        assert len(rows.rows) == 3

        scoring_functions = [
            "basic::equality",
        ]
        task_id = "meta-reference::app_eval"
        await eval_tasks_impl.register_eval_task(
            eval_task_id=task_id,
            dataset_id="test_dataset_for_eval",
            scoring_functions=scoring_functions,
        )
        response = await eval_impl.evaluate_rows(
            task_id=task_id,
            input_rows=rows.rows,
            scoring_functions=scoring_functions,
            task_config=AppEvalTaskConfig(
                eval_candidate=ModelCandidate(
                    model=inference_model,
                    sampling_params=SamplingParams(),
                ),
                scoring_params={
                    "meta-reference::llm_as_judge_base": LLMAsJudgeScoringFnParams(
                        judge_model=judge_model,
                        prompt_template=JUDGE_PROMPT,
                        judge_score_regexes=[
                            r"Total rating: (\d+)",
                            r"rating: (\d+)",
                            r"Rating: (\d+)",
                        ],
                    )
                },
            ),
        )
        assert len(response.generations) == 3
        assert "basic::equality" in response.scores

    @pytest.mark.asyncio
    async def test_eval_run_eval(self, eval_stack, inference_model, judge_model):
        eval_impl, eval_tasks_impl, datasets_impl, models_impl = (
            eval_stack[Api.eval],
            eval_stack[Api.eval_tasks],
            eval_stack[Api.datasets],
            eval_stack[Api.models],
        )

        await register_dataset(
            datasets_impl, for_generation=True, dataset_id="test_dataset_for_eval"
        )

        scoring_functions = [
            "basic::subset_of",
        ]

        task_id = "meta-reference::app_eval-2"
        await eval_tasks_impl.register_eval_task(
            eval_task_id=task_id,
            dataset_id="test_dataset_for_eval",
            scoring_functions=scoring_functions,
        )
        response = await eval_impl.run_eval(
            task_id=task_id,
            task_config=AppEvalTaskConfig(
                eval_candidate=ModelCandidate(
                    model=inference_model,
                    sampling_params=SamplingParams(),
                ),
            ),
        )
        assert response.job_id == "0"
        job_status = await eval_impl.job_status(task_id, response.job_id)
        assert job_status and job_status.value == "completed"
        eval_response = await eval_impl.job_result(task_id, response.job_id)

        assert eval_response is not None
        assert len(eval_response.generations) == 5
        assert "basic::subset_of" in eval_response.scores

    @pytest.mark.asyncio
    async def test_eval_run_benchmark_eval(self, eval_stack, inference_model):
        eval_impl, eval_tasks_impl, datasets_impl, models_impl = (
            eval_stack[Api.eval],
            eval_stack[Api.eval_tasks],
            eval_stack[Api.datasets],
            eval_stack[Api.models],
        )

        response = await datasets_impl.list_datasets()
        assert len(response) > 0
        if response[0].provider_id != "huggingface":
            pytest.skip(
                "Only huggingface provider supports pre-registered remote datasets"
            )

        await datasets_impl.register_dataset(
            dataset_id="mmlu",
            dataset_schema={
                "input_query": StringType(),
                "expected_answer": StringType(),
                "chat_completion_input": ChatCompletionInputType(),
            },
            url=URL(uri="https://huggingface.co/datasets/llamastack/evals"),
            metadata={
                "path": "llamastack/evals",
                "name": "evals__mmlu__details",
                "split": "train",
            },
        )

        # register eval task
        await eval_tasks_impl.register_eval_task(
            eval_task_id="meta-reference-mmlu",
            dataset_id="mmlu",
            scoring_functions=["basic::regex_parser_multiple_choice_answer"],
        )

        # list benchmarks
        response = await eval_tasks_impl.list_eval_tasks()
        assert len(response) > 0

        benchmark_id = "meta-reference-mmlu"
        response = await eval_impl.run_eval(
            task_id=benchmark_id,
            task_config=BenchmarkEvalTaskConfig(
                eval_candidate=ModelCandidate(
                    model=inference_model,
                    sampling_params=SamplingParams(),
                ),
                num_examples=3,
            ),
        )
        job_status = await eval_impl.job_status(benchmark_id, response.job_id)
        assert job_status and job_status.value == "completed"
        eval_response = await eval_impl.job_result(benchmark_id, response.job_id)
        assert eval_response is not None
        assert len(eval_response.generations) == 3
