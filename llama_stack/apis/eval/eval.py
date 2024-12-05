# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Optional, Protocol, Union

from typing_extensions import Annotated

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.schema_utils import json_schema_type, webmethod
from llama_stack.apis.scoring_functions import *  # noqa: F403
from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.eval_tasks import *  # noqa: F403


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


EvalCandidate = Annotated[
    Union[ModelCandidate, AgentCandidate], Field(discriminator="type")
]


@json_schema_type
class BenchmarkEvalTaskConfig(BaseModel):
    type: Literal["benchmark"] = "benchmark"
    eval_candidate: EvalCandidate
    num_examples: Optional[int] = Field(
        description="Number of examples to evaluate (useful for testing), if not provided, all examples in the dataset will be evaluated",
        default=None,
    )


@json_schema_type
class AppEvalTaskConfig(BaseModel):
    type: Literal["app"] = "app"
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


EvalTaskConfig = Annotated[
    Union[BenchmarkEvalTaskConfig, AppEvalTaskConfig], Field(discriminator="type")
]


@json_schema_type
class EvaluateResponse(BaseModel):
    generations: List[Dict[str, Any]]
    # each key in the dict is a scoring function name
    scores: Dict[str, ScoringResult]


class Eval(Protocol):
    @webmethod(route="/eval/run-eval", method="POST")
    async def run_eval(
        self,
        task_id: str,
        task_config: EvalTaskConfig,
    ) -> Job: ...

    @webmethod(route="/eval/evaluate-rows", method="POST")
    async def evaluate_rows(
        self,
        task_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        task_config: EvalTaskConfig,
    ) -> EvaluateResponse: ...

    @webmethod(route="/eval/job/status", method="GET")
    async def job_status(self, task_id: str, job_id: str) -> Optional[JobStatus]: ...

    @webmethod(route="/eval/job/cancel", method="POST")
    async def job_cancel(self, task_id: str, job_id: str) -> None: ...

    @webmethod(route="/eval/job/result", method="GET")
    async def job_result(self, task_id: str, job_id: str) -> EvaluateResponse: ...
