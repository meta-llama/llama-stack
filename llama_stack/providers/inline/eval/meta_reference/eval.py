# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, List, Optional
from llama_models.llama3.api.datatypes import *  # noqa: F403
from tqdm import tqdm

from .....apis.common.job_types import Job
from .....apis.eval.eval import Eval, EvalTaskConfig, EvaluateResponse, JobStatus
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.agents import Agents
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.eval_tasks import EvalTask
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import Scoring
from llama_stack.providers.datatypes import EvalTasksProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl

from .config import MetaReferenceEvalConfig

EVAL_TASKS_PREFIX = "eval_tasks:"


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
        agents_api: Agents,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.scoring_api = scoring_api
        self.inference_api = inference_api
        self.agents_api = agents_api

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}

        self.eval_tasks = {}

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing eval_tasks from kvstore
        start_key = EVAL_TASKS_PREFIX
        end_key = f"{EVAL_TASKS_PREFIX}\xff"
        stored_eval_tasks = await self.kvstore.range(start_key, end_key)

        for eval_task in stored_eval_tasks:
            eval_task = EvalTask.model_validate_json(eval_task)
            self.eval_tasks[eval_task.identifier] = eval_task

    async def shutdown(self) -> None: ...

    async def register_eval_task(self, task_def: EvalTask) -> None:
        # Store in kvstore
        key = f"{EVAL_TASKS_PREFIX}{task_def.identifier}"
        await self.kvstore.set(
            key=key,
            value=task_def.model_dump_json(),
        )
        self.eval_tasks[task_def.identifier] = task_def

    async def validate_eval_input_dataset_schema(self, dataset_id: str) -> None:
        dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)
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

    async def run_eval(
        self,
        task_id: str,
        task_config: EvalTaskConfig,
    ) -> Job:
        task_def = self.eval_tasks[task_id]
        dataset_id = task_def.dataset_id
        candidate = task_config.eval_candidate
        scoring_functions = task_def.scoring_functions

        await self.validate_eval_input_dataset_schema(dataset_id=dataset_id)
        all_rows = await self.datasetio_api.get_rows_paginated(
            dataset_id=dataset_id,
            rows_in_page=(
                -1 if task_config.num_examples is None else task_config.num_examples
            ),
        )
        res = await self.evaluate_rows(
            task_id=task_id,
            input_rows=all_rows.rows,
            scoring_functions=scoring_functions,
            task_config=task_config,
        )

        # TODO: currently needs to wait for generation before returning
        # need job scheduler queue (ray/celery) w/ jobs api
        job_id = str(len(self.jobs))
        self.jobs[job_id] = res
        return Job(job_id=job_id)

    async def _run_agent_generation(
        self, input_rows: List[Dict[str, Any]], task_config: EvalTaskConfig
    ) -> List[Dict[str, Any]]:
        candidate = task_config.eval_candidate
        create_response = await self.agents_api.create_agent(candidate.config)
        agent_id = create_response.agent_id

        generations = []
        for i, x in tqdm(enumerate(input_rows)):
            assert ColumnName.chat_completion_input.value in x, "Invalid input row"
            input_messages = eval(str(x[ColumnName.chat_completion_input.value]))
            input_messages = [UserMessage(**x) for x in input_messages]

            # NOTE: only single-turn agent generation is supported. Create a new session for each input row
            session_create_response = await self.agents_api.create_agent_session(
                agent_id, f"session-{i}"
            )
            session_id = session_create_response.session_id

            turn_request = dict(
                agent_id=agent_id,
                session_id=session_id,
                messages=input_messages,
                stream=True,
            )
            turn_response = [
                chunk
                async for chunk in await self.agents_api.create_agent_turn(
                    **turn_request
                )
            ]
            final_event = turn_response[-1].event.payload
            generations.append(
                {
                    ColumnName.generated_answer.value: final_event.turn.output_message.content
                }
            )

        return generations

    async def _run_model_generation(
        self, input_rows: List[Dict[str, Any]], task_config: EvalTaskConfig
    ) -> List[Dict[str, Any]]:
        candidate = task_config.eval_candidate
        assert (
            candidate.sampling_params.max_tokens is not None
        ), "SamplingParams.max_tokens must be provided"

        generations = []
        for x in tqdm(input_rows):
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
                    model_id=candidate.model,
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

        return generations

    async def evaluate_rows(
        self,
        task_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: EvalTaskConfig,
    ) -> EvaluateResponse:
        candidate = task_config.eval_candidate
        if candidate.type == "agent":
            generations = await self._run_agent_generation(input_rows, task_config)
        elif candidate.type == "model":
            generations = await self._run_model_generation(input_rows, task_config)
        else:
            raise ValueError(f"Invalid candidate type: {candidate.type}")

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

    async def job_status(self, task_id: str, job_id: str) -> Optional[JobStatus]:
        if job_id in self.jobs:
            return JobStatus.completed

        return None

    async def job_cancel(self, task_id: str, job_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, task_id: str, job_id: str) -> EvaluateResponse:
        status = await self.job_status(task_id, job_id)
        if not status or status != JobStatus.completed:
            raise ValueError(f"Job is not completed, Status: {status.value}")

        return self.jobs[job_id]
