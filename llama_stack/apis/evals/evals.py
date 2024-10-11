# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Protocol

from llama_models.schema_utils import webmethod
from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.dataset import *  # noqa: F403


class EvaluationJob(BaseModel):
    job_uuid: str


class EvaluationJobLogStream(BaseModel):
    job_uuid: str


@json_schema_type
class EvalResult(BaseModel):
    """Evaluation result."""

    metrics: Dict[str, str]


@json_schema_type
class SingleEvalResult(BaseModel):
    """Single evaluation result."""

    score_data: Dict[str, float]


@json_schema_type
class EvaluateResponse(BaseModel):
    """Scores for evaluation."""

    eval_result: EvalResult
    formatted_report: Optional[str] = None


@json_schema_type
class EvaluationJobStatusResponse(BaseModel):
    job_uuid: str


@json_schema_type
class EvaluationJobArtifactsResponse(BaseModel):
    """Artifacts of a evaluation job."""

    job_uuid: str


@json_schema_type
class EvaluationJobCreateResponse(BaseModel):
    """Response to create a evaluation job."""

    job_uuid: str


@json_schema_type
class EvaluateTaskConfig(BaseModel):
    # num examples to evaluate, evaluate all if None
    n_samples: Optional[int] = None
    # model evaluation params
    sampling_params: SamplingParams = SamplingParams()


class BaseTask(
    ABC,
    Generic[
        TDatasetSample,
        TPreprocessedSample,
        TPredictionSample,
        TPostprocessedSample,
        TSingleEvalResult,
    ],
):
    """
    A task represents a single evaluation benchmark, including it's dataset, preprocessing, postprocessing and scoring methods.
    Base class for all evaluation tasks. Each task needs to implement the following methods:
    - F1: preprocess_sample(self)
    - F2: postprocess_sample(self)
    - F3: score_sample(self)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = self.__class__.__name__

    @abstractmethod
    def preprocess_sample(self, sample: TDatasetSample) -> TPreprocessedSample:
        raise NotImplementedError()

    @abstractmethod
    def postprocess_sample(self, sample: TPredictionSample) -> TPostprocessedSample:
        raise NotImplementedError()

    @abstractmethod
    def score_sample(
        self, sample: TPostprocessedSample, ground_truth: TPreprocessedSample
    ):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        raise NotImplementedError()

    def preprocess(
        self, dataset: BaseDataset[TDatasetSample]
    ) -> List[TPreprocessedSample]:
        return [self.preprocess_sample(sample) for sample in self.dataset]

    def postprocess(
        self, generation: List[TPredictionSample]
    ) -> List[TPostprocessedSample]:
        return [self.postprocess_sample(sample) for sample in generation]

    def score(
        self,
        postprocessed: List[TPostprocessedSample],
        preprocessed_dataset: List[TPreprocessedSample],
    ) -> List[TSingleEvalResult]:
        return [
            self.score_sample(sample, ground_truth)
            for sample, ground_truth in zip(postprocessed, self.preprocessed_dataset)
        ]


class Evals(Protocol):
    @webmethod(route="/evals/run")
    async def run_evals(
        self,
        model: str,
        task: str,
        dataset: Optional[str] = None,
        eval_task_config: Optional[EvaluateTaskConfig] = None,
    ) -> EvaluateResponse: ...

    # @webmethod(route="/evals/jobs")
    # def get_evaluation_jobs(self) -> List[EvaluationJob]: ...

    # @webmethod(route="/evals/job/create")
    # async def create_evaluation_job(
    #     self, model: str, dataset: str, task: str
    # ) -> EvaluationJob: ...

    # @webmethod(route="/evals/job/status")
    # def get_evaluation_job_status(
    #     self, job_uuid: str
    # ) -> EvaluationJobStatusResponse: ...

    # # sends SSE stream of logs
    # @webmethod(route="/evals/job/logs")
    # def get_evaluation_job_logstream(self, job_uuid: str) -> EvaluationJobLogStream: ...

    # @webmethod(route="/evals/job/cancel")
    # def cancel_evaluation_job(self, job_uuid: str) -> None: ...

    # @webmethod(route="/evals/job/artifacts")
    # def get_evaluation_job_artifacts(
    #     self, job_uuid: str
    # ) -> EvaluationJobArtifactsResponse: ...
