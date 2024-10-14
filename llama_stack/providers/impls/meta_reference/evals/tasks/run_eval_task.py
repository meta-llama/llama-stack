# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.distribution.registry.datasets.dataset_registry import DatasetRegistry

from llama_stack.apis.evals import *  # noqa: F403


class RunEvalTask(BaseTask):
    """
    RunEvalTask for LlamaStack
    """

    def __init__(
        self,
        eval_task_config,
        generator_processor: Optional[BaseGeneratorProcessor] = None,
        generator: Optional[BaseGenerator] = None,
        scorer: Optional[BaseScorer] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            generator_processor=generator_processor,
            generator=generator,
            scorer=scorer,
            *args,
            **kwargs,
        )
        self.eval_task_config = eval_task_config
        self.dataset = DatasetRegistry.get_dataset(
            eval_task_config.dataset_config.dataset_name
        )

    def run(self, *args, **kwargs) -> EvalResult:
        print(f"Running eval task on {self.dataset}")
        return EvalResult()
