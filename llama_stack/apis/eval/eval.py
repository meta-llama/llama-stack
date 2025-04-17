# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import BaseJob
from llama_stack.apis.inference import SamplingParams, SystemMessage
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


class EvaluateJob(BaseJob, BaseModel):
    type: Literal["eval"] = "eval"


class ListEvaluateJobsResponse(BaseModel):
    data: list[EvaluateJob]


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


class Eval(Protocol):
    """Llama Stack Evaluation API for running evaluations on model and agent candidates."""

    @webmethod(route="/eval/benchmarks/{benchmark_id}/evaluate", method="POST")
    async def evaluate(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateJob: ...

    # CRUD operations on running jobs
    @webmethod(route="/evaluate/jobs/{job_id:path}", method="GET")
    async def get_evaluate_job(self, job_id: str) -> EvaluateJob: ...

    @webmethod(route="/evaluate/jobs", method="GET")
    async def list_evaluate_jobs(self) -> ListEvaluateJobsResponse: ...

    @webmethod(route="/evaluate/jobs/{job_id:path}", method="POST")
    async def update_evaluate_job(self, job: EvaluateJob) -> EvaluateJob: ...

    @webmethod(route="/evaluate/job/{job_id:path}", method="DELETE")
    async def delete_evaluate_job(self, job_id: str) -> None: ...

    # Note: pause/resume/cancel are achieved as follows:
    # - POST with status=paused
    # - POST with status=resuming
    # - POST with status=cancelled
