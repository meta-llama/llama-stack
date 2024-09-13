# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Generator, List, Optional

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message, ToolPromptFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.sku_list import resolve_model

from .config import MetaReferenceImplConfig
from .generation import Llama, model_checkpoint_dir
from .parallel_utils import ModelParallelProcessGroup


@dataclass
class InferenceArgs:
    messages: List[Message]
    temperature: float
    top_p: float
    max_gen_len: int
    logprobs: bool
    tool_prompt_format: ToolPromptFormat


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    # the `task` object is the same that is sent to `ModelParallelProcessGroup.run_inference()`
    def __call__(self, task: InferenceArgs):
        return self.llama.chat_completion(
            task.messages,
            task.temperature,
            task.top_p,
            task.max_gen_len,
            task.logprobs,
            task.tool_prompt_format,
        )


def init_model_cb(config: MetaReferenceImplConfig):
    llama = Llama.build(config)
    return ModelRunner(llama)


class LlamaModelParallelGenerator:
    """
    This abstraction exists so
     - we can run model parallel code without needing to run the CLIs via torchrun
     - this also enables use model parallel code within a notebook context.

    A Context Manager is used to ensure that the model parallel process is started and stopped
    correctly. This does make the ergonomics a little awkward, because it isn't immediately
    clear at the callsite why we need to use a context manager.
    """

    def __init__(self, config: MetaReferenceImplConfig):
        self.config = config
        self.model = resolve_model(self.config.model)
        # this is a hack because Agent's loop uses this to tokenize and check if input is too long
        # while the tool-use loop is going
        checkpoint_dir = model_checkpoint_dir(self.model)
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
        self.formatter = ChatFormat(Tokenizer(tokenizer_path))

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        self.group = ModelParallelProcessGroup(
            self.config.model_parallel_size,
            init_model_cb=partial(init_model_cb, self.config),
        )
        self.group.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.group.stop()

    def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ) -> Generator:
        req_obj = InferenceArgs(
            messages=deepcopy(messages),
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            tool_prompt_format=tool_prompt_format,
        )

        gen = self.group.run_inference(req_obj)
        yield from gen
