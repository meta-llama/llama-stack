# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json

from termcolor import cprint

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403

from .config import MetaReferenceEvalsImplConfig
from .tasks.run_eval_task import RunEvalTask
from .tasks.run_scoring_task import RunScoringTask


class MetaReferenceEvalsImpl(Evals):
    def __init__(self, config: MetaReferenceEvalsImplConfig, inference_api: Inference):
        self.inference_api = inference_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_eval_task(
        self,
        eval_task_config: EvaluateTaskConfig,
    ) -> EvaluateResponse:
        cprint(f"run_eval_task: on {eval_task_config}", "green")

        run_task = RunEvalTask()
        eval_result = await run_task.run(eval_task_config, self.inference_api)

        return EvaluateResponse(
            eval_result=eval_result,
            formatted_report=json.dumps(eval_result.json(), indent=4),
        )

    async def run_scorer(
        self,
        dataset_config: EvaluateDatasetConfig,
        eval_scoring_config: EvaluateScoringConfig,
    ) -> EvaluateResponse:
        cprint(f"run_scorer: on {dataset_config} with {eval_scoring_config}", "green")

        run_task = RunScoringTask()
        eval_result = await run_task.run(dataset_config, eval_scoring_config)

        return EvaluateResponse(
            eval_result=eval_result,
            formatted_report=json.dumps(eval_result.json(), indent=4),
        )
