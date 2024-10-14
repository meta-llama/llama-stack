# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.distribution.registry.datasets import DatasetRegistry
from llama_stack.providers.impls.meta_reference.evals.scorer.basic_scorers import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.generator.inference_generator import (
    InferenceGenerator,
)
from llama_stack.providers.impls.meta_reference.evals.processor.mmlu_processor import (
    MMLUProcessor,
)

from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from termcolor import cprint


class RunEvalTask(BaseTask):
    """
    RunEvalTask for LlamaStack
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    async def run(
        self,
        eval_task_config: EvaluateTaskConfig,
        inference_api: Inference,
        *args,
        **kwargs,
    ) -> EvalResult:
        print(f"Running eval task w/ {eval_task_config}")

        print(DatasetRegistry.names())
        dataset = DatasetRegistry.get(eval_task_config.dataset_config.dataset_name)
        dataset.load(n_samples=eval_task_config.dataset_config.row_limit)
        print(f"Running on {len(dataset)} samples")

        # F1
        processor = MMLUProcessor()
        preprocessed = processor.preprocess(dataset)

        # Generation
        generator = InferenceGenerator(
            model=eval_task_config.generation_config.model,
            inference_api=inference_api,
        )
        generation_outputs = await generator.generate(preprocessed)

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

        return eval_result
