# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.agents import Agents
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.inference import Inference
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

from .....apis.benchmarks import Benchmark
from .....apis.evaluation.evaluation import (
    Evaluation,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationResponse,
    EvaluationTask,
)
from .config import MetaReferenceEvaluationConfig

EVAL_TASKS_PREFIX = "benchmarks:"


class MetaReferenceEvaluationImpl(
    Evaluation,
    BenchmarksProtocolPrivate,
):
    def __init__(
        self,
        config: MetaReferenceEvaluationConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
        inference_api: Inference,
        agents_api: Agents,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.inference_api = inference_api
        self.agents_api = agents_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        pass

    async def run(
        self,
        task: EvaluationTask,
        candidate: EvaluationCandidate,
    ) -> EvaluationJob:
        raise NotImplementedError("Run is not implemented yet")

    async def run_sync(
        self,
        task: EvaluationTask,
        candidate: EvaluationCandidate,
    ) -> EvaluationResponse:
        raise NotImplementedError("Run sync is not implemented yet")

    async def grade(self, task: EvaluationTask) -> EvaluationJob:
        raise NotImplementedError("Grade is not implemented yet")

    async def grade_sync(self, task: EvaluationTask) -> EvaluationResponse:
        raise NotImplementedError("Grade sync is not implemented yet")
