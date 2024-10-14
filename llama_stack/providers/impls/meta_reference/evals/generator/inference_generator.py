# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from termcolor import cprint

from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403


class InferenceGenerator(BaseGenerator[PreprocessedSample, GenerationResponseSample]):
    """
    InferenceGenerator for LlamaStack
    """

    def __init__(
        self,
        model,
        inference_api,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.inference_api = inference_api

    async def generate(
        self, preprocessed_dataset: List[PreprocessedSample]
    ) -> List[GenerationResponseSample]:
        generation_outputs = []
        for sample in preprocessed_dataset:
            print("generation: ", sample)
            response = await self.inference_api.chat_completion(
                model=self.model,
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
        return generation_outputs
