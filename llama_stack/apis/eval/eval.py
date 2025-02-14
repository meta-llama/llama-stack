# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.apis.scoring import ScoringResult
from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class ModelCandidate(BaseModel):
    type: Literal["model"] = "model"
    model: str
    sampling_params: SamplingParams
    system_message: Optional[SystemMessage] = None


@json_schema_type
class AgentCandidate(BaseModel):
    type: Literal["agent"] = "agent"
    config: AgentConfig


EvalCandidate = register_schema(
    Annotated[Union[ModelCandidate, AgentCandidate], Field(discriminator="type")],
    name="EvalCandidate",
)


@json_schema_type
class BenchmarkConfig(BaseModel):
    type: Literal["benchmark"] = "benchmark"
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
    generations: List[Dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: Dict[str, ScoringResult]


class Eval(Protocol):
    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs", method="POST")
    async def run_eval(
        self,
        benchmark_id: str,
        task_config: BenchmarkConfig,
    ) -> Job: ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/evaluations", method="POST")
    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: BenchmarkConfig,
    ) -> EvaluateResponse: ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="GET")
    async def job_status(self, benchmark_id: str, job_id: str) -> Optional[JobStatus]: ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}", method="DELETE")
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None: ...

    @webmethod(route="/eval/benchmarks/{benchmark_id}/jobs/{job_id}/result", method="GET")
    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse: ...

    @webmethod(route="/eval/tasks/{task_id}/jobs", method="POST")
    async def DEPRECATED_run_eval(
        self,
        task_id: str,
        task_config: BenchmarkConfig,
    ) -> Job: ...

    @webmethod(route="/eval/tasks/{task_id}/evaluations", method="POST")
    async def DEPRECATED_evaluate_rows(
        self,
        task_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: BenchmarkConfig,
    ) -> EvaluateResponse: ...

    @webmethod(route="/eval/tasks/{task_id}/jobs/{job_id}", method="GET")
    async def DEPRECATED_job_status(self, task_id: str, job_id: str) -> Optional[JobStatus]: ...

    @webmethod(route="/eval/tasks/{task_id}/jobs/{job_id}", method="DELETE")
    async def DEPRECATED_job_cancel(self, task_id: str, job_id: str) -> None: ...

    @webmethod(route="/eval/tasks/{task_id}/jobs/{job_id}/result", method="GET")
    async def DEPRECATED_job_result(self, task_id: str, job_id: str) -> EvaluateResponse: ...
