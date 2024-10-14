# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json

from termcolor import cprint

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.dataset import *  # noqa: F403

from .config import MetaReferenceEvalsImplConfig

# from llama_stack.distribution.registry.tasks.task_registry import TaskRegistry
from .tasks.run_eval_task import RunEvalTask


class MetaReferenceEvalsImpl(Evals):
    def __init__(self, config: MetaReferenceEvalsImplConfig, inference_api: Inference):
        self.inference_api = inference_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_eval_task(
        self,
        model: str,
        task: str,
        dataset: Optional[str] = None,
        eval_task_config: Optional[EvaluateTaskConfig] = None,
    ) -> EvaluateResponse:
        cprint(
            f"model={model}, dataset={dataset}, task={task}, eval_task_config={eval_task_config}",
            "red",
        )

        if not dataset:
            raise ValueError("dataset must be specified for mete-reference evals")

        if not eval_task_config:
            # construct eval task config from inputs
            eval_task_config = EvaluateTaskConfig(
                dataset_config=EvaluateDatasetConfig(
                    dataset_name=dataset,
                    row_limit=2,
                ),
                generation_config=EvaluateModelGenerationConfig(
                    model=model,
                ),
                scoring_config=EvaluateScoringConfig(
                    scorer_config_list=[
                        EvaluateSingleScorerConfig(scorer_name="accuracy"),
                    ]
                ),
            )

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
        return EvaluateResponse(
            eval_result={},
        )
