# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import CommonJobFields
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


EvalCandidate = register_schema(
    Annotated[Union[ModelCandidate, AgentCandidate], Field(discriminator="type")],
    name="EvalCandidate",
)


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


@json_schema_type
class EvalJob(CommonJobFields):
    """The EvalJob object representing a evaluation job that was created through API."""

    type: Literal["eval"] = "eval"
    # TODO: result files or result datasets ids?
    result_files: List[str] = Field(
        default_factory=list,
        description="Result files of an evaluation run. Which can be queried for results.",
    )


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
        :param candidate: Candidate to evaluate on.
            - {
                "type": "model",
                "model": "Llama-3.1-8B-Instruct",
                "sampling_params": {...},
                "system_message": "You are a helpful assistant.",
            }
            - {
                "type": "agent",
                "config": {...},
            }
        :return: The job that was created to run the evaluation.
        """

    @webmethod(route="/eval/rows", method="POST")
    async def evaluate_rows(
        self,
        dataset_rows: List[Dict[str, Any]],
        scoring_functions: List[ScoringFnParams],
        candidate: EvalCandidate,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a candidate.

        :param dataset_rows: The rows to evaluate.
        :param scoring_functions: The scoring functions to use for the evaluation.
        :param candidate: The candidate to evaluate on.
        :return: EvaluateResponse object containing generations and scores
        """

    @webmethod(route="/eval/jobs/{job_id}", method="GET")
    async def get_job(self, benchmark_id: str, job_id: str) -> Optional[EvalJob]:
        """Get the EvalJob object for a given job id and benchmark id.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to get the status of.
        :return: EvalJob object indicating its status
        """
        ...

    @webmethod(route="/eval/jobs/{job_id}", method="DELETE")
    async def cancel_job(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job.

        :param benchmark_id: The ID of the benchmark to run the evaluation on.
        :param job_id: The ID of the job to cancel.
        """
        ...
