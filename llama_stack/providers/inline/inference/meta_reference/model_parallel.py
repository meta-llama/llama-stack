# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from copy import deepcopy
from functools import partial
from typing import Any, Generator

from llama_stack.models.llama.datatypes import Model
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.providers.utils.inference.prompt_adapter import (
    ChatCompletionRequestWithRawContent,
    CompletionRequestWithRawContent,
)

from .common import model_checkpoint_dir
from .config import MetaReferenceInferenceConfig
from .llama3.generation import Llama3
from .parallel_utils import ModelParallelProcessGroup


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    # the `task` object is the same that is sent to `ModelParallelProcessGroup.run_inference()`
    def __call__(self, req: Any):
        if isinstance(req, ChatCompletionRequestWithRawContent):
            return self.llama.chat_completion(req)
        elif isinstance(req, CompletionRequestWithRawContent):
            return self.llama.completion(req)
        else:
            raise ValueError(f"Unexpected task type {type(req)}")


def init_model_cb(
    config: MetaReferenceInferenceConfig,
    model_id: str,
    llama_model: Model,
):
    llama = Llama3.build(config, model_id, llama_model)
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

    def __init__(
        self,
        config: MetaReferenceInferenceConfig,
        model_id: str,
        llama_model: Model,
    ):
        self.config = config
        self.model_id = model_id
        self.llama_model = llama_model

        # this is a hack because Agent's loop uses this to tokenize and check if input is too long
        # while the tool-use loop is going
        resolved_model = resolve_model(model_id)
        if resolved_model is None:
            # if the model is not a native llama model, get the default checkpoint_dir based on model id
            checkpoint_dir = model_checkpoint_dir(model_id)
        else:
            # if the model is a native llama model, get the default checkpoint_dir based on model core_model_id value
            checkpoint_dir = model_checkpoint_dir(resolved_model.descriptor())
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
        self.formatter = ChatFormat(Tokenizer(tokenizer_path))

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        model_parallel_size = self.llama_model.pth_file_count

        self.group = ModelParallelProcessGroup(
            model_parallel_size,
            init_model_cb=partial(init_model_cb, self.config, self.model_id, self.llama_model),
        )
        self.group.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.group.stop()

    def completion(
        self,
        request: CompletionRequestWithRawContent,
    ) -> Generator:
        req_obj = deepcopy(request)
        gen = self.group.run_inference(req_obj)
        yield from gen

    def chat_completion(
        self,
        request: ChatCompletionRequestWithRawContent,
    ) -> Generator:
        req_obj = deepcopy(request)
        gen = self.group.run_inference(req_obj)
        yield from gen
