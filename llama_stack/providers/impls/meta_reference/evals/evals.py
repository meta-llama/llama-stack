# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from termcolor import cprint

from llama_stack.providers.impls.meta_reference.evals.datas.dataset_registry import (
    get_dataset,
)
from llama_stack.providers.impls.meta_reference.evals.tasks.task_registry import (
    get_task,
)

from .config import MetaReferenceEvalsImplConfig


class MetaReferenceEvalsImpl(Evals):
    def __init__(self, config: MetaReferenceEvalsImplConfig, inference_api: Inference):
        self.inference_api = inference_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_evals(
        self,
        model: str,
        dataset: str,
        task: str,
    ) -> EvaluateResponse:
        cprint(f"model={model}, dataset={dataset}, task={task}", "red")
        dataset = get_dataset(dataset)
        task_impl = get_task(task, dataset)
        x1 = task_impl.preprocess()

        # TODO: replace w/ batch inference & async return eval job
        generation_outputs = []
        for msg in x1[:5]:
            response = self.inference_api.chat_completion(
                model=model,
                messages=[msg],
                stream=False,
            )
            async for x in response:
                generation_outputs.append(x.completion_message.content)

        x2 = task_impl.postprocess(generation_outputs)
        eval_results = task_impl.score(x2)
        eval_response = task_impl.aggregate_results(eval_results)
        return eval_response
