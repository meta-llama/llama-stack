# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
from termcolor import cprint

from llama_stack.providers.impls.meta_reference.evals.datas.utils import (  # noqa: F403
    get_dataset,
)
from llama_stack.providers.impls.meta_reference.evals.tasks.utils import (  # noqa: F403
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

        # resolve dataset
        # - either a custom URL dataset or HF URL dataset
        dataset = get_dataset("mmlu_eval")
        print(dataset.dataset)

        # # resolve task and execute task
        task_impl = get_task(task, dataset)
        print(task_impl)

        # # F1: this will generate a preprocessed list of input messages for model
        # x1 = task_impl.preprocess(dataset)

        # # call inference API w/ model
        # generation_outputs = ["response1", "response2", "response3"]

        # # F2: post process
        # x2 = task_impl.postprocess(generation_outputs)

        # # F3: score generation outputs
        # scores = task_impl.score(x2)

        return EvaluateResponse(
            metrics={
                "accuracy": 0.5,
            }
        )
