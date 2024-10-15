# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
import uuid
from typing import Any

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .config import VLLMConfig


log = logging.getLogger(__name__)


def _random_uuid() -> str:
    return str(uuid.uuid4().hex)


def _vllm_sampling_params(sampling_params: Any) -> SamplingParams:
    """Convert sampling params to vLLM sampling params."""
    if sampling_params is None:
        return SamplingParams()

    # TODO convert what I saw in my first test ... but surely there's more to do here
    kwargs = {
        "temperature": sampling_params.temperature,
    }
    if sampling_params.top_k >= 1:
        kwargs["top_k"] = sampling_params.top_k
    if sampling_params.top_p:
        kwargs["top_p"] = sampling_params.top_p
    if sampling_params.max_tokens >= 1:
        kwargs["max_tokens"] = sampling_params.max_tokens
    if sampling_params.repetition_penalty > 0:
        kwargs["repetition_penalty"] = sampling_params.repetition_penalty

    return SamplingParams(**kwargs)


class VLLMInferenceImpl(ModelRegistryHelper, Inference):
    """Inference implementation for vLLM."""

    HF_MODEL_MAPPINGS = {
        # TODO: seems like we should be able to build this table dynamically ...
        "Llama3.1-8B": "meta-llama/Llama-3.1-8B",
        "Llama3.1-70B": "meta-llama/Llama-3.1-70B",
        "Llama3.1-405B:bf16-mp8": "meta-llama/Llama-3.1-405B",
        "Llama3.1-405B": "meta-llama/Llama-3.1-405B-FP8",
        "Llama3.1-405B:bf16-mp16": "meta-llama/Llama-3.1-405B",
        "Llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Llama3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
        "Llama3.1-405B-Instruct:bf16-mp8": "meta-llama/Llama-3.1-405B-Instruct",
        "Llama3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct-FP8",
        "Llama3.1-405B-Instruct:bf16-mp16": "meta-llama/Llama-3.1-405B-Instruct",
        "Llama3.2-1B": "meta-llama/Llama-3.2-1B",
        "Llama3.2-3B": "meta-llama/Llama-3.2-3B",
        "Llama3.2-11B-Vision": "meta-llama/Llama-3.2-11B-Vision",
        "Llama3.2-90B-Vision": "meta-llama/Llama-3.2-90B-Vision",
        "Llama3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "Llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "Llama3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Llama3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision",
        "Llama-Guard-3-1B:int4-mp1": "meta-llama/Llama-Guard-3-1B-INT4",
        "Llama-Guard-3-1B": "meta-llama/Llama-Guard-3-1B",
        "Llama-Guard-3-8B": "meta-llama/Llama-Guard-3-8B",
        "Llama-Guard-3-8B:int8-mp1": "meta-llama/Llama-Guard-3-8B-INT8",
        "Prompt-Guard-86M": "meta-llama/Prompt-Guard-86M",
        "Llama-Guard-2-8B": "meta-llama/Llama-Guard-2-8B",
    }

    def __init__(self, config: VLLMConfig):
        Inference.__init__(self)
        ModelRegistryHelper.__init__(
            self,
            stack_to_provider_models_map=self.HF_MODEL_MAPPINGS,
        )
        self.config = config
        self.engine = None

        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    async def initialize(self):
        """Initialize the vLLM inference adapter."""

        log.info("Initializing vLLM inference adapter")

        # Disable usage stats reporting. This would be a surprising thing for most
        # people to find out was on by default.
        # https://docs.vllm.ai/en/latest/serving/usage_stats.html
        if "VLLM_NO_USAGE_STATS" not in os.environ:
            os.environ["VLLM_NO_USAGE_STATS"] = "1"

        hf_model = self.HF_MODEL_MAPPINGS.get(self.config.model)

        # TODO -- there are a ton of options supported here ...
        engine_args = AsyncEngineArgs()
        engine_args.model = hf_model
        # We will need a new config item for this in the future if model support is more broad
        # than it is today (llama only)
        engine_args.tokenizer = hf_model
        engine_args.tensor_parallel_size = self.config.tensor_parallel_size

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def shutdown(self):
        """Shutdown the vLLM inference adapter."""
        log.info("Shutting down vLLM inference adapter")
        if self.engine:
            self.engine.shutdown_background_loop()

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Any | None = ...,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | CompletionResponseStreamChunk:
        log.info("vLLM completion")
        messages = [UserMessage(content=content)]
        return self.chat_completion(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        )

    def chat_completion(
        self,
        model: str,
        messages: list[Message],
        sampling_params: Any | None = ...,
        tools: list[ToolDefinition] | None = ...,
        tool_choice: ToolChoice | None = ...,
        tool_prompt_format: ToolPromptFormat | None = ...,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> ChatCompletionResponse | ChatCompletionResponseStreamChunk:
        log.info("vLLM chat completion")

        assert self.engine is not None

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        log.info("Sampling params: %s", sampling_params)
        request_id = _random_uuid()

        prompt = chat_completion_request_to_prompt(request, self.formatter)
        vllm_sampling_params = _vllm_sampling_params(request.sampling_params)
        results_generator = self.engine.generate(
            prompt, vllm_sampling_params, request_id
        )
        if stream:
            return self._stream_chat_completion(request, results_generator)
        else:
            return self._nonstream_chat_completion(request, results_generator)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, results_generator: AsyncGenerator
    ) -> ChatCompletionResponse:
        outputs = [o async for o in results_generator]
        final_output = outputs[-1]

        assert final_output is not None
        outputs = final_output.outputs
        finish_reason = outputs[-1].stop_reason
        choice = OpenAICompatCompletionChoice(
            finish_reason=finish_reason,
            text="".join([output.text for output in outputs]),
        )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )
        return process_chat_completion_response(response, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, results_generator: AsyncGenerator
    ) -> AsyncGenerator:
        async def _generate_and_convert_to_openai_compat():
            async for chunk in results_generator:
                if not chunk.outputs:
                    log.warning("Empty chunk received")
                    continue

                text = "".join([output.text for output in chunk.outputs])
                choice = OpenAICompatCompletionChoice(
                    finish_reason=chunk.outputs[-1].stop_reason,
                    text=text,
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    async def embeddings(
        self, model: str, contents: list[InterleavedTextMedia]
    ) -> EmbeddingsResponse:
        log.info("vLLM embeddings")
        # TODO
        raise NotImplementedError()
