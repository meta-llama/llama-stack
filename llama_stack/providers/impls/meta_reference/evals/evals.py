# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from termcolor import cprint

from llama_stack.distribution.registry.datasets.dataset_registry import DatasetRegistry

from llama_stack.distribution.registry.tasks.task_registry import TaskRegistry

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
        dataset = DatasetRegistry.get_dataset(dataset)
        dataset.load()
        task_impl = TaskRegistry.get_task(task)(dataset)

        x1 = task_impl.preprocess()

        # TODO: replace w/ batch inference & async return eval job
        generation_outputs = []
        print("generation start")
        for msg in x1[:5]:
            print("generation for msg: ", msg)
            response = await self.inference_api.chat_completion(
                model=model,
                messages=[msg],
                stream=False,
            )
            generation_outputs.append(response.completion_message.content)

        x2 = task_impl.postprocess(generation_outputs)
        eval_results = task_impl.score(x2)
        eval_response = task_impl.aggregate_results(eval_results)
        return eval_response
