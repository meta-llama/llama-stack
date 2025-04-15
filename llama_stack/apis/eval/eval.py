# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import Job
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.apis.scoring import ScoringResult
from llama_stack.apis.scoring_functions import ScoringFnParams
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


EvalCandidate = Annotated[Union[ModelCandidate, AgentCandidate], Field(discriminator="type")]
register_schema(EvalCandidate, name="EvalCandidate")


@json_schema_type
class BenchmarkConfig(BaseModel):
    """A benchmark configuration for evaluation.

    :param eval_candidate: The candidate to evaluate.
    :param scoring_params: Map between scoring function id and parameters for each scoring function you want to run
    :param num_examples: (Optional) The number of examples to evaluate. If not provided, all examples in the dataset will be evaluated
    """

    eval_candidate: EvalCandidate
    scoring_params: Dict[str, ScoringFnParams] = Field(
        description="Map between scoring function id and parameters for each scoring function you want to run",
        default_factory=dict,
    )
    num_examples: Optional[int] = Field(
        description="Number of examples to evaluate (useful for testing), if not provided, all examples in the dataset will be evaluated",
        default=None,
    )
    # we could optinally add any specific dataset config here


@json_schema_type
class EvaluateResponse(BaseModel):
    """The response from an evaluation.

    :param generations: The generations from the evaluation.
    :param scores: The scores from the evaluation.
    """

    generations: List[Dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: Dict[str, ScoringResult]


class Eval(Protocol):
    """Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs", method="POST")
    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Run an evaluation on a benchmark.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param benchmark_config: The configuration for the benchmark.
        :return: The job that was created to run the evaluation.
        """

    @webmethod(route="/eval/benchmarks/{benchmark_id}/evaluations", method="POST")
    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param input_rows: The rows to evaluate.
        :param scoring_functions: The scoring functions to use for the evaluation.
        :param benchmark_config: The configuration for the benchmark.
        :return: EvaluateResponse object containing generations and scores
        """

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="GET")
    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to get the status of.
        :return: The status of the evaluationjob.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="DELETE")
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to cancel.
        """
        ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}/result", method="GET")
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the result of a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to get the result of.
        :return: The result of the job.
        """
