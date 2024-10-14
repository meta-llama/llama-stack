# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json

from termcolor import cprint

from llama_stack.providers.impls.meta_reference.evals.scorer.basic_scorers import (
    AggregateScorer,
)

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.dataset import *  # noqa: F403

from llama_stack.distribution.registry.datasets.dataset_registry import DatasetRegistry
from llama_stack.providers.impls.meta_reference.evals.processor.mmlu_processor import (
    MMLUProcessor,
)

# from llama_stack.distribution.registry.tasks.task_registry import TaskRegistry
# from .tasks.run_eval_task import RunEvalTask
from .scorer.basic_scorers import *  # noqa: F403


from .config import MetaReferenceEvalsImplConfig


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

        # TODO: wrap inside task
        # run_task = RunEvalTask(
        #     eval_task_config=eval_task_config,
        # )
        # eval_result = run_task.run()

        dataset = DatasetRegistry.get_dataset(
            eval_task_config.dataset_config.dataset_name
        )
        dataset.load(n_samples=eval_task_config.dataset_config.row_limit)
        print(f"Running on {len(dataset)} samples")

        # F1
        processor = MMLUProcessor()
        preprocessed = processor.preprocess(dataset)

        # Generation
        # TODO: wrap inside BaseGenerator
        generation_outputs = []
        for sample in preprocessed:
            print("generation: ", sample)
            response = await self.inference_api.chat_completion(
                model=model,
                messages=sample.generation_input.messages,
                stream=False,
            )
            cprint(f"response: {response}", "cyan")

            generation_outputs.append(
                GenerationResponseSample(
                    generation_output=GenerationOutput(
                        completion_message=response.completion_message.content
                    )
                )
            )
        cprint(generation_outputs, "green")

        # F2
        postprocessed = processor.postprocess(generation_outputs, dataset)
        cprint(postprocessed, "blue")

        # F3 - scorer
        scorer = AggregateScorer(
            scorers=[
                AccuracyScorer(),
                RandomScorer(),
            ]
        )

        scorer_results = scorer.score(postprocessed)
        cprint(scorer_results, "magenta")
        eval_result = scorer.aggregate_results(scorer_results)

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
