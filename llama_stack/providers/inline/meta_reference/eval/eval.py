# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from llama_models.llama3.api.datatypes import *  # noqa: F403

from .....apis.common.job_types import Job
from .....apis.eval.eval import (
    AppEvalTaskConfig,
    BenchmarkEvalTaskConfig,
    Eval,
    EvalTaskConfig,
    EvaluateResponse,
    JobStatus,
)
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.eval_tasks import EvalTaskDef
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import Scoring
from llama_stack.providers.datatypes import EvalTasksProtocolPrivate

from .config import MetaReferenceEvalConfig


# NOTE: this is the default eval task identifier for app eval
# it is used to make the router work for all app evals
# For app eval using other eval providers, the eval task identifier will be different
DEFAULT_EVAL_TASK_IDENTIFIER = "meta-reference::app_eval"


class ColumnName(Enum):
    input_query = "input_query"
    expected_answer = "expected_answer"
    chat_completion_input = "chat_completion_input"
    completion_input = "completion_input"
    generated_answer = "generated_answer"


class MetaReferenceEvalImpl(Eval, EvalTasksProtocolPrivate):
    def __init__(
        self,
        config: MetaReferenceEvalConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        scoring_api: Scoring,
        inference_api: Inference,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.scoring_api = scoring_api
        self.inference_api = inference_api

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def list_eval_tasks(self) -> List[EvalTaskDef]:
        # NOTE: In order to be routed to this provider, the eval task def must have
        # a EvalTaskDef with identifier defined as DEFAULT_EVAL_TASK_IDENTIFIER
        # for app eval where eval task benchmark_id is not pre-registered
        eval_tasks = [
            EvalTaskDef(
                identifier=DEFAULT_EVAL_TASK_IDENTIFIER,
                dataset_id="",
                scoring_functions=[],
            )
        ]
        return eval_tasks

    async def validate_eval_input_dataset_schema(self, dataset_id: str) -> None:
        dataset_def = await self.datasets_api.get_dataset(dataset_identifier=dataset_id)
        if not dataset_def.dataset_schema or len(dataset_def.dataset_schema) == 0:
            raise ValueError(f"Dataset {dataset_id} does not have a schema defined.")

        expected_schemas = [
            {
                ColumnName.input_query.value: StringType(),
                ColumnName.expected_answer.value: StringType(),
                ColumnName.chat_completion_input.value: ChatCompletionInputType(),
            },
            {
                ColumnName.input_query.value: StringType(),
                ColumnName.expected_answer.value: StringType(),
                ColumnName.completion_input.value: CompletionInputType(),
            },
        ]

        if dataset_def.dataset_schema not in expected_schemas:
            raise ValueError(
                f"Dataset {dataset_id} does not have a correct input schema in {expected_schemas}"
            )

    async def run_benchmark(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkEvalTaskConfig,
    ) -> Job:
        raise NotImplementedError("Benchmark eval is not implemented yet")

    async def run_eval(
        self,
        task: EvalTaskDef,
        task_config: AppEvalTaskConfig,
    ) -> Job:
        dataset_id = task.dataset_id
        candidate = task_config.eval_candidate
        scoring_functions = task.scoring_functions

        await self.validate_eval_input_dataset_schema(dataset_id=dataset_id)
        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        res = await self.evaluate_rows(
            input_rows=all_rows.rows,
            scoring_functions=scoring_functions,
            task_config=task_config,
        )

        # TODO: currently needs to wait for generation before returning
        # need job scheduler queue (ray/celery) w/ jobs api
        job_id = str(len(self.jobs))
        self.jobs[job_id] = res
        return Job(job_id=job_id)

    async def evaluate_rows(
        self,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: EvalTaskConfig,
        eval_task_id: Optional[str] = None,
    ) -> EvaluateResponse:
        candidate = task_config.eval_candidate
        if candidate.type == "agent":
            raise NotImplementedError(
                "Evaluation with generation has not been implemented for agents"
            )
        assert (
            candidate.sampling_params.max_tokens is not None
        ), "SamplingParams.max_tokens must be provided"

        generations = []
        for x in input_rows:
            if ColumnName.completion_input.value in x:
                input_content = eval(str(x[ColumnName.completion_input.value]))
                response = await self.inference_api.completion(
                    model=candidate.model,
                    content=input_content,
                    sampling_params=candidate.sampling_params,
                )
                generations.append(
                    {
                        ColumnName.generated_answer.value: response.completion_message.content
                    }
                )
            elif ColumnName.chat_completion_input.value in x:
                chat_completion_input_str = str(
                    x[ColumnName.chat_completion_input.value]
                )
                input_messages = eval(chat_completion_input_str)
                input_messages = [UserMessage(**x) for x in input_messages]
                messages = []
                if candidate.system_message:
                    messages.append(candidate.system_message)
                messages += input_messages
                response = await self.inference_api.chat_completion(
                    model=candidate.model,
                    messages=messages,
                    sampling_params=candidate.sampling_params,
                )
                generations.append(
                    {
                        ColumnName.generated_answer.value: response.completion_message.content
                    }
                )
            else:
                raise ValueError("Invalid input row")

        # scoring with generated_answer
        score_input_rows = [
            input_r | generated_r
            for input_r, generated_r in zip(input_rows, generations)
        ]

        if task_config.type == "app" and task_config.scoring_params is not None:
            scoring_functions_dict = {
                scoring_fn_id: task_config.scoring_params.get(scoring_fn_id, None)
                for scoring_fn_id in scoring_functions
            }
        else:
            scoring_functions_dict = {
                scoring_fn_id: None for scoring_fn_id in scoring_functions
            }

        score_response = await self.scoring_api.score(
            input_rows=score_input_rows, scoring_functions=scoring_functions_dict
        )

        return EvaluateResponse(generations=generations, scores=score_response.results)

    async def job_status(self, job_id: str, eval_task_id: str) -> Optional[JobStatus]:
        if job_id in self.jobs:
            return JobStatus.completed

        return None

    async def job_cancel(self, job_id: str, eval_task_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, job_id: str, eval_task_id: str) -> EvaluateResponse:
        status = await self.job_status(job_id, eval_task_id)
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job is not completed, Status: {status.value}")

        return self.jobs[job_id]
