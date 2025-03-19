# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import CommonJobFields, JobType
from llama_stack.apis.datasets import DataSource
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation.

    :param model: The model ID to evaluate.
    :param sampling_params: The sampling parameters for the model.
    :param system_message: (Optional) The system message providing instructions or context to the model.
    """

    type: Literal["model"] = "model"
    model_id: str
    sampling_params: SamplingParams
    system_message: Optional[SystemMessage] = None


@json_schema_type
class AgentCandidate(BaseModel):
    """An agent candidate for evaluation.

    :param config: The configuration for the agent candidate.
    """

    type: Literal["agent"] = "agent"
    agent_config: AgentConfig


EvaluationCandidate = register_schema(
    Annotated[Union[ModelCandidate, AgentCandidate], Field(discriminator="type")],
    name="EvaluationCandidate",
)


@json_schema_type
class EvaluationTask(BaseModel):
    """
    A task for evaluation. To specify a task, one of the following must be provided:
    - `benchmark_id`: Run evaluation task against a benchmark_id. Use this when you have a curated dataset and have settled on the graders.
    - `dataset_id` and `grader_ids`: Run evaluation task against a dataset_id and a list of grader_ids. Use this when you have datasets and / or are iterating on your graders. 
    - `data_source` and `grader_ids`: Run evaluation task against a data source (e.g. rows, uri, etc.) and a list of grader_ids. Prefer this when you are early in your evaluation cycle and experimenting much more with your data and graders.

    :param benchmark_id: The benchmark ID to evaluate.
    :param dataset_id: The dataset ID to evaluate.
    :param data_source: The data source to evaluate.
    :param grader_ids: The grader IDs to evaluate.
    """

    benchmark_id: Optional[str] = None
    dataset_id: Optional[str] = None
    data_source: Optional[DataSource] = None
    grader_ids: Optional[List[str]] = None


@json_schema_type
class EvaluationJob(CommonJobFields):
    type: Literal[JobType.evaluation.value] = JobType.evaluation.value

    # input params for the submitted evaluation job
    task: EvaluationTask
    candidate: EvaluationCandidate


@json_schema_type
class ScoringResult(BaseModel):
    """
    A scoring result for a single row.

    :param scores: The scoring result for each row. Each row is a map of grader column name to value.
    :param metrics: Map of metric name to aggregated value.
    """

    scores: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@json_schema_type
class EvaluationResponse(BaseModel):
    """
    A response to an inline evaluation.

    :param generations: The generations in rows for the evaluation.
    :param scores: The scores for the evaluation. Map of grader id to ScoringResult.
    """

    generations: List[Dict[str, Any]]
    scores: Dict[str, ScoringResult]


class Evaluation(Protocol):
    @webmethod(route="/evaluation/run", method="POST")
    async def run(
        self,
        task: EvaluationTask,
        candidate: EvaluationCandidate,
    ) -> EvaluationJob:
        """
        Schedule a full evaluation job, by generating results using candidate and grading them.

        :param task: The task to evaluate. To specify a task, one of the following must be provided:
         - `benchmark_id`: Run evaluation task against a benchmark_id
         - `dataset_id` and `grader_ids`: Run evaluation task against a dataset_id and a list of grader_ids
         - `data_source` and `grader_ids`: Run evaluation task against a data source (e.g. rows, uri, etc.) and a list of grader_ids
        :param candidate: The candidate to evaluate.
        """
        ...

    @webmethod(route="/evaluation/run_sync", method="POST")
    async def run_sync(
        self,
        task: EvaluationTask,
        candidate: EvaluationCandidate,
    ) -> EvaluationResponse:
        """
        Run an evaluation synchronously, i.e., without scheduling a job".
        You should use this for quick testing, or when the number of rows is limited. Some implementations may have stricter restrictions on inputs which will be accepted.

        :param task: The task to evaluate. To specify a task, one of the following must be provided:
         - `benchmark_id`: Run evaluation task against a benchmark_id
         - `dataset_id` and `grader_ids`: Run evaluation task against a dataset_id and a list of grader_ids
         - `data_source` and `grader_ids`: Run evaluation task against a data source (e.g. rows, uri, etc.) and a list of grader_ids
        :param candidate: The candidate to evaluate.
        """
        ...

    @webmethod(route="/evaluation/grade", method="POST")
    async def grade(self, task: EvaluationTask) -> EvaluationJob:
        """
        Schedule a grading job, by grading generated (model or agent) results. The generated results are expected to be in the dataset.

        :param task: The task to evaluate. To specify a task, one of the following must be provided:
         - `benchmark_id`: Run evaluation task against a benchmark_id
         - `dataset_id` and `grader_ids`: Run evaluation task against a dataset_id and a list of grader_ids
         - `data_source` and `grader_ids`: Run evaluation task against a data source (e.g. rows, uri, etc.) and a list of grader_ids

        :return: The evaluation job containing grader scores.
        """
        ...

    @webmethod(route="/evaluation/grade_sync", method="POST")
    async def grade_sync(self, task: EvaluationTask) -> EvaluationResponse:
        """
        Run grading synchronously on generated results, i.e., without scheduling a job.
        You should use this for quick testing, or when the number of rows is limited. Some implementations may have stricter restrictions on inputs which will be accepted.

        :param task: The task to evaluate. To specify a task, one of the following must be provided:
         - `benchmark_id`: Run evaluation task against a benchmark_id
         - `dataset_id` and `grader_ids`: Run evaluation task against a dataset_id and a list of grader_ids
         - `data_source` and `grader_ids`: Run evaluation task against a data source (e.g. rows, uri, etc.) and a list of grader_ids

        :return: The evaluation job containing grader scores. "generations" is not populated in the response.
        """
        ...
