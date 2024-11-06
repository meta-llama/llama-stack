# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from copy import deepcopy
from functools import partial
from typing import Any, Generator

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.sku_list import resolve_model

from llama_stack.apis.inference import ChatCompletionRequest, CompletionRequest

from .config import MetaReferenceInferenceConfig
from .generation import Llama, model_checkpoint_dir
from .parallel_utils import ModelParallelProcessGroup


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    # the `task` object is the same that is sent to `ModelParallelProcessGroup.run_inference()`
    def __call__(self, req: Any):
        if isinstance(req, ChatCompletionRequest):
            return self.llama.chat_completion(req)
        elif isinstance(req, CompletionRequest):
            return self.llama.completion(req)
        else:
            raise ValueError(f"Unexpected task type {type(req)}")


def init_model_cb(config: MetaReferenceInferenceConfig):
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

    def __init__(self, config: MetaReferenceInferenceConfig):
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

    def completion(
        self,
        request: CompletionRequest,
    ) -> Generator:
        req_obj = deepcopy(request)
        gen = self.group.run_inference(req_obj)
        yield from gen

    def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Generator:
        req_obj = deepcopy(request)
        gen = self.group.run_inference(req_obj)
        yield from gen
