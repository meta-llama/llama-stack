# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.evals import *  # noqa: F403

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
        print("hi")
        return EvaluateResponse(
            metrics={
                "accuracy": 0.5,
            }
        )
