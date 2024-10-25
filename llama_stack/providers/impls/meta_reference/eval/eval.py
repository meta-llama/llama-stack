# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.common.job_types import Job
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.eval import Eval, EvalCandidate, EvaluateResponse, JobStatus
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import Scoring

from .config import MetaReferenceEvalConfig


class ColumnName(Enum):
    input_query = "input_query"
    expected_answer = "expected_answer"
    chat_completion_input = "chat_completion_input"
    completion_input = "completion_input"
    generated_answer = "generated_answer"


class MetaReferenceEvalImpl(Eval):
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

    async def evaluate_batch(
        self,
        dataset_id: str,
        candidate: EvalCandidate,
        scoring_functions: List[str],
    ) -> Job:
        await self.validate_eval_input_dataset_schema(dataset_id=dataset_id)
        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=-1,
        )
        res = await self.evaluate(
            input_rows=all_rows.rows,
            candidate=candidate,
            scoring_functions=scoring_functions,
        )

        # TODO: currently needs to wait for generation before returning
        # need job scheduler queue (ray/celery) w/ jobs api
        job_id = str(len(self.jobs))
        self.jobs[job_id] = res
        return Job(job_id=job_id)

    async def evaluate(
        self,
        input_rows: List[Dict[str, Any]],
        candidate: EvalCandidate,
        scoring_functions: List[str],
    ) -> EvaluateResponse:
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
                input_messages = eval(str(x[ColumnName.chat_completion_input.value]))
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

        score_response = await self.scoring_api.score(
            input_rows=score_input_rows, scoring_functions=scoring_functions
        )

        return EvaluateResponse(generations=generations, scores=score_response.results)

    async def job_status(self, job_id: str) -> Optional[JobStatus]:
        if job_id in self.jobs:
            return JobStatus.completed

        return None

    async def job_cancel(self, job_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, job_id: str) -> EvaluateResponse:
        status = await self.job_status(job_id)
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job is not completed, Status: {status.value}")

        return self.jobs[job_id]
