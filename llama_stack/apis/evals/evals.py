# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Protocol

from llama_models.schema_utils import webmethod
from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403


class EvaluationJob(BaseModel):
    job_uuid: str


class EvaluationJobLogStream(BaseModel):
    job_uuid: str


@json_schema_type
class EvalResult(BaseModel):
    """Aggregated final evaluation result."""

    metrics: Dict[str, float]


@json_schema_type
class SingleEvalResult(BaseModel):
    """Single evaluation result. Contains a scorer name, and corresponding metrics from scorer."""

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
class EvaluateDatasetConfig(BaseModel):
    # identifier to previously registered dataset via DatasetDef
    dataset_name: str
    # limit number of rows to evaluate
    row_limit: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = None


@json_schema_type
class EvaluatePreprocessConfig(BaseModel):
    kwargs: Optional[Dict[str, Any]] = None


@json_schema_type
class EvaluateModelGenerationConfig(BaseModel):
    model: str
    sampling_params: SamplingParams = SamplingParams()
    kwargs: Optional[Dict[str, Any]] = None


@json_schema_type
class EvaluatePostprocessConfig(BaseModel):
    kwargs: Optional[Dict[str, Any]] = None


@json_schema_type
class EvaluateJudgeScoringConfig(BaseModel): ...


@json_schema_type
class LLMJudgeConfig(BaseModel):
    judge_preprocess_config: EvaluatePreprocessConfig
    judge_model_generation_config: EvaluateModelGenerationConfig
    judge_postprocess_config: EvaluatePostprocessConfig
    judge_scoring_config: EvaluateJudgeScoringConfig


@json_schema_type
class EvaluateSingleScorerConfig(BaseModel):
    scorer_name: str
    llm_judge_config: Optional[LLMJudgeConfig] = None


@json_schema_type
class EvaluateScoringConfig(BaseModel):
    # list of scorer (metrics) names to use
    scorer_config_list: List[EvaluateSingleScorerConfig]


@json_schema_type
class EvaluateTaskConfig(BaseModel):
    dataset_config: EvaluateDatasetConfig
    preprocess_config: Optional[EvaluatePreprocessConfig] = None
    generation_config: EvaluateModelGenerationConfig
    postprocess_config: Optional[EvaluatePostprocessConfig] = None
    scoring_config: EvaluateScoringConfig


class BaseGeneratorProcessor(
    ABC,
    Generic[
        TDatasetSample,
        TPreprocessedSample,
        TGenerationResponseSample,
        TScorerInputSample,
    ],
):
    """
    Base class for all generator processors. Each processor needs to implement the following methods:
    - F1: preprocess_sample(self, dataset)
    - F2: postprocess_sample(self)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    def preprocess(
        self, dataset: BaseDataset[TDatasetSample]
    ) -> List[TPreprocessedSample]:
        return [self.preprocess_sample(sample) for sample in dataset]

    def postprocess(
        self,
        generation: List[TGenerationResponseSample],
        dataset: BaseDataset[TDatasetSample],
    ) -> List[TScorerInputSample]:
        return [
            self.postprocess_sample(generation_sample, dataset_sample)
            for generation_sample, dataset_sample in zip(generation, dataset)
        ]

    @abstractmethod
    def preprocess_sample(self, sample: TDatasetSample) -> TPreprocessedSample:
        raise NotImplementedError()

    @abstractmethod
    def postprocess_sample(
        self,
        generation_sample: TGenerationResponseSample,
        dataset_sample: TDatasetSample,
    ) -> TScorerInputSample:
        raise NotImplementedError()


class BaseGenerator(ABC, Generic[TPreprocessedSample, TGenerationResponseSample]):
    """
    Base class for all generators. Each generator needs to implement the following methods:
    - generate(self, preprocessed_dataset)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def generate(
        self, preprocessed_dataset: List[TPreprocessedSample]
    ) -> List[TGenerationResponseSample]:
        raise NotImplementedError()


class BaseScorer(ABC, Generic[TScorerInputSample]):
    """
    Base class for all scorers. Each scorer needs to implement the following methods:
    - score_sample(self, scorer_input_sample)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def score_sample(self, scorer_input_sample: TScorerInputSample) -> SingleEvalResult:
        raise NotImplementedError()

    @abstractmethod
    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        raise NotImplementedError()

    def score(
        self, prepared_eval_dataset: List[TScorerInputSample]
    ) -> List[SingleEvalResult]:
        return [self.score_sample(sample) for sample in prepared_eval_dataset]


class BaseTask(ABC):
    def __init__(
        self,
        generator_processor: Optional[BaseGeneratorProcessor] = None,
        generator: Optional[BaseGenerator] = None,
        scorer: Optional[BaseScorer] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generator_processor = generator_processor
        self.generator = generator
        self.scorer = scorer

    @abstractmethod
    async def run(self, *args, **kwargs) -> EvalResult:
        raise NotImplementedError()


class Evals(Protocol):

    @webmethod(route="/evals/run_eval_task")
    async def run_eval_task(
        self,
        model: str,
        task: str,
        dataset: Optional[str] = None,
        eval_task_config: Optional[EvaluateTaskConfig] = None,
    ) -> EvaluateResponse: ...

    @webmethod(route="/evals/run_scorer")
    async def run_scorer(
        self,
        dataset_config: EvaluateDatasetConfig,
        eval_scoring_config: EvaluateScoringConfig,
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
