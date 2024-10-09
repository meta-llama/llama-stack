# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403
import os
import random
from pathlib import Path

import lm_eval
from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate, get_task_list
from lm_eval.tasks import get_task_dict, TaskManager
from termcolor import cprint

from .config import EleutherEvalsImplConfig  # noqa


class EleutherEvalsWrapper(LM):
    def __init__(
        self,
        inference_api: Inference,
        model: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_api = inference_api
        self.model = model
        self.tokenizer = None
        self.tokenized_requests = False
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        raise NotImplementedError("Not implemented")

    @property
    def max_length(self) -> int:
        return NotImplementedError("Not implemented")

    @property
    def max_gen_toks(self) -> int:
        return NotImplementedError("Not implemented")

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str) -> List[int]:
        return NotImplementedError("Not implemented")

    def tok_decode(self, tokens: List[int]) -> str:
        return NotImplementedError("Not implemented")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        # TODO: implement inference completion with loglikelihood
        res = []
        for req in requests:
            res.append((-random.random(), False))

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        res = []
        for req in requests:
            response = self.inference_api.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": req.args[0],
                    }
                ],
                stream=False,
            )
            print(response)
            res.append(response.completion_message.content)

        return res


class EleutherEvalsAdapter(Evals):
    def __init__(self, config: EleutherEvalsImplConfig, inference_api: Inference):
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
        eluther_wrapper = EleutherEvalsWrapper(self.inference_api, model)

        cprint(f"Eleuther Evals: {model} {dataset} {task}", "red")

        task = "meta_mmlu_pro_instruct"
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        print(current_dir)

        task_manager = TaskManager(
            include_path=str(current_dir / "tasks"),
        )

        task_dict = get_task_dict(task, task_manager)
        cprint(task_dict, "blue")

        task_types = set([t.task.OUTPUT_TYPE for t in get_task_list(task_dict)])
        cprint(task_types, "cyan")

        output = evaluate(
            eluther_wrapper,
            task_dict,
            limit=10,
        )

        formatted_output = lm_eval.utils.make_table(output)

        cprint(formatted_output, "green")

        return EvaluateResponse(
            metrics={
                "metrics_table": formatted_output,
            },
        )
