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
from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    InterleavedTextMedia,
    Message,
    StopReason,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_models.llama3.api.tokenizer import Tokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from llama_stack.apis.inference import ChatCompletionRequest, Inference

from llama_stack.apis.inference.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    LogProbConfig,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.providers.utils.inference.augment_messages import (
    augment_messages_for_tools,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

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

    return SamplingParams().from_optional(**kwargs)


class VLLMInferenceImpl(Inference, ModelRegistryHelper):
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

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Any | None = ...,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | CompletionResponseStreamChunk:
        log.info("vLLM completion")
        messages = [Message(role="user", content=content)]
        async for result in self.chat_completion(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        ):
            yield result

    async def chat_completion(
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
        vllm_sampling_params = _vllm_sampling_params(sampling_params)

        messages = augment_messages_for_tools(request)
        log.info("Augmented messages: %s", messages)
        prompt = "".join([str(message.content) for message in messages])

        request_id = _random_uuid()
        results_generator = self.engine.generate(
            prompt, vllm_sampling_params, request_id
        )

        if not stream:
            # Non-streaming case
            final_output = None
            stop_reason = None
            async for request_output in results_generator:
                final_output = request_output
                if stop_reason is None and request_output.outputs:
                    reason = request_output.outputs[-1].stop_reason
                    if reason == "stop":
                        stop_reason = StopReason.end_of_turn
                    elif reason == "length":
                        stop_reason = StopReason.out_of_tokens

            if not stop_reason:
                stop_reason = StopReason.end_of_message

            if final_output:
                response = "".join([output.text for output in final_output.outputs])
                yield ChatCompletionResponse(
                    completion_message=CompletionMessage(
                        content=response,
                        stop_reason=stop_reason,
                    ),
                    logprobs=None,
                )
        else:
            # Streaming case
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )

            buffer = ""
            last_chunk = ""
            ipython = False
            stop_reason = None

            async for chunk in results_generator:
                if not chunk.outputs:
                    log.warning("Empty chunk received")
                    continue

                if chunk.outputs[-1].stop_reason:
                    reason = chunk.outputs[-1].stop_reason
                    if stop_reason is None and reason == "stop":
                        stop_reason = StopReason.end_of_turn
                    elif stop_reason is None and reason == "length":
                        stop_reason = StopReason.out_of_tokens
                    break

                text = "".join([output.text for output in chunk.outputs])

                # check if its a tool call ( aka starts with <|python_tag|> )
                if not ipython and text.startswith("<|python_tag|>"):
                    ipython = True
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content="",
                                parse_status=ToolCallParseStatus.started,
                            ),
                        )
                    )
                    buffer += text
                    continue

                if ipython:
                    if text == "<|eot_id|>":
                        stop_reason = StopReason.end_of_turn
                        text = ""
                        continue
                    elif text == "<|eom_id|>":
                        stop_reason = StopReason.end_of_message
                        text = ""
                        continue

                    buffer += text
                    delta = ToolCallDelta(
                        content=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                        )
                    )
                else:
                    last_chunk_len = len(last_chunk)
                    last_chunk = text
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=text[last_chunk_len:],
                            stop_reason=stop_reason,
                        )
                    )

            if not stop_reason:
                stop_reason = StopReason.end_of_message

            # parse tool calls and report errors
            message = self.formatter.decode_assistant_message_from_content(
                buffer, stop_reason
            )
            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content="",
                            parse_status=ToolCallParseStatus.failure,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content=tool_call,
                            parse_status=ToolCallParseStatus.success,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta="",
                    stop_reason=stop_reason,
                )
            )

    async def embeddings(
        self, model: str, contents: list[InterleavedTextMedia]
    ) -> EmbeddingsResponse:
        log.info("vLLM embeddings")
        # TODO
        raise NotImplementedError()
