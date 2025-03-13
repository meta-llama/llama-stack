# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import CommonJobFields, JobStatus
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.apis.scoring import ScoringResult
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class ModelCandidate(BaseModel):
    """A model candidate for evaluation.

    :param model: The model ID to evaluate.
    :param sampling_params: The sampling parameters for the model.
    :param system_message: (Optional) The system message providing instructions or context to the model.
    """

    type: Literal["model"] = "model"
    model: str
    sampling_params: SamplingParams
    system_message: Optional[SystemMessage] = None


@json_schema_type
class AgentCandidate(BaseModel):
    """An agent candidate for evaluation.

    :param config: The configuration for the agent candidate.
    """

    type: Literal["agent"] = "agent"
    config: AgentConfig


EvalCandidate = register_schema(
    Annotated[Union[ModelCandidate, AgentCandidate], Field(discriminator="type")],
    name="EvalCandidate",
)


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation.

    :param generations: The generations from the evaluation.
    :param scores: The scores from the evaluation.
    """

    generations: List[Dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: Dict[str, ScoringResult]


@json_schema_type
class EvalJob(CommonJobFields):
    type: Literal["eval"] = "eval"
    result_files: List[str] = Field(
        description="The file ids of the eval results.",
        default_factory=list,
    )
    result_datasets: List[str] = Field(
        description="The ids of the datasets containing the eval results.",
        default_factory=list,
    )

    # how the job is created
    benchmark_id: str = Field(description="The id of the benchmark to evaluate on.")
    candidate: EvalCandidate = Field(description="The candidate to evaluate on.")


class Eval(Protocol):
    """Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    @webmethod(route="/eval/jobs", method="POST")
    async def evaluate_benchmark(
        self,
        benchmark_id: str,
        candidate: EvalCandidate,
    ) -> EvalJob:
        """Run an evaluation on a benchmark.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param candidate: The candidate to evaluate on.
        :return: The job that was created to run the evaluation.
        """

    @webmethod(route="/eval/rows", method="POST")
    async def evaluate_rows(
        self,
        dataset_rows: List[Dict[str, Any]],
        scoring_fn_ids: List[str],
        candidate: EvalCandidate,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a candidate.

        :param dataset_rows: The rows to evaluate.
        :param scoring_fn_ids: The scoring function ids to use for the evaluation.
        :param candidate: The candidate to evaluate on.
        :return: EvaluateResponse object containing generations and scores
        """

    @webmethod(route="/eval/jobs", method="GET")
    async def list_eval_jobs(self) -> List[EvalJob]:
        """List all evaluation jobs.

        :return: A list of evaluation jobs.
        """
        ...

    @webmethod(route="/eval/jobs/{job_id}", method="GET")
    async def get_eval_job(self, job_id: str) -> Optional[EvalJob]:
        """Get a job by id.

        :param job_id: The id of the job to get.
        :return: The job.
        """
        ...

    @webmethod(route="/eval/jobs/{job_id}", method="DELETE")
    async def delete_eval_job(self, job_id: str) -> Optional[EvalJob]:
        """Delete a job.

        :param job_id: The id of the job to delete.
        """
        ...

    @webmethod(route="/eval/jobs/{job_id}/cancel", method="POST")
    async def cancel_eval_job(self, job_id: str) -> Optional[EvalJob]:
        """Cancel a job.

        :param job_id: The id of the job to cancel.
        """
        ...
