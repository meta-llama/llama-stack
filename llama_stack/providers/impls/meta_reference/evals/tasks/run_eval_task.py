# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.distribution.registry.datasets import DatasetRegistry
from llama_stack.distribution.registry.generator_processors import (
    GeneratorProcessorRegistry,
)
from llama_stack.distribution.registry.scorers import ScorerRegistry

from llama_stack.providers.impls.meta_reference.evals.scorer.aggregate_scorer import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.scorer.basic_scorers import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.generator.inference_generator import (
    InferenceGenerator,
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
        dataset = DatasetRegistry.get(
            eval_task_config.dataset_config.dataset_identifier
        )
        dataset.load(n_samples=eval_task_config.dataset_config.row_limit)
        print(f"Running on {len(dataset)} samples")

        # F1
        print(GeneratorProcessorRegistry.names())
        processor = GeneratorProcessorRegistry.get(
            eval_task_config.processor_config.processor_identifier
        )()
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
        scorer_config_list = eval_task_config.scoring_config.scorer_config_list
        scorer_list = []
        for s_conf in scorer_config_list:
            scorer = ScorerRegistry.get(s_conf.scorer_name)
            if s_conf.llm_judge_config:
                scorer_list.append(
                    scorer(
                        llm_judge_config=s_conf.llm_judge_config,
                        inference_api=inference_api,
                    )
                )
            else:
                scorer_list.append(scorer())

        scorer = AggregateScorer(
            scorers=scorer_list,
        )

        scorer_results = scorer.score(postprocessed)
        cprint(scorer_results, "magenta")
        eval_result = scorer.aggregate_results(scorer_results)

        return eval_result
