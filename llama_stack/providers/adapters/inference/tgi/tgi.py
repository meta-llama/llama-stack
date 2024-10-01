# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
from typing import AsyncGenerator

from huggingface_hub import AsyncInferenceClient, HfApi
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.distribution.datatypes import RoutableProvider

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.augment_messages import (
    augment_messages_for_tools,
)

from .config import InferenceAPIImplConfig, InferenceEndpointImplConfig, TGIImplConfig

logger = logging.getLogger(__name__)


class _HfAdapter(Inference, RoutableProvider):
    client: AsyncInferenceClient
    max_tokens: int
    model_id: str

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)

    async def validate_routing_keys(self, routing_keys: list[str]) -> None:
        # these are the model names the Llama Stack will use to route requests to this provider
        # perform validation here if necessary
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    def get_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
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

        messages = augment_messages_for_tools(request)
        model_input = self.formatter.encode_dialog_prompt(messages)
        prompt = self.tokenizer.decode(model_input.tokens)

        input_tokens = len(model_input.tokens)
        max_new_tokens = min(
            request.sampling_params.max_tokens or (self.max_tokens - input_tokens),
            self.max_tokens - input_tokens - 1,
        )

        print(f"Calculated max_new_tokens: {max_new_tokens}")

        options = self.get_chat_options(request)
        if not request.stream:
            response = await self.client.text_generation(
                prompt=prompt,
                stream=False,
                details=True,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            )
            stop_reason = None
            if response.details.finish_reason:
                if response.details.finish_reason in ["stop", "eos_token"]:
                    stop_reason = StopReason.end_of_turn
                elif response.details.finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                response.generated_text,
                stop_reason,
            )
            yield ChatCompletionResponse(
                completion_message=completion_message,
                logprobs=None,
            )

        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )
            buffer = ""
            ipython = False
            stop_reason = None
            tokens = []

            async for response in await self.client.text_generation(
                prompt=prompt,
                stream=True,
                details=True,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            ):
                token_result = response.token

                buffer += token_result.text
                tokens.append(token_result.id)

                if not ipython and buffer.startswith("<|python_tag|>"):
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
                    buffer = buffer[len("<|python_tag|>") :]
                    continue

                if token_result.text == "<|eot_id|>":
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.text == "<|eom_id|>":
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                if ipython:
                    delta = ToolCallDelta(
                        content=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )
                else:
                    delta = text

                if stop_reason is None:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                        )
                    )

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            # parse tool calls and report errors
            message = self.formatter.decode_assistant_message(tokens, stop_reason)
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


class TGIAdapter(_HfAdapter):
    async def initialize(self, config: TGIImplConfig) -> None:
        self.client = AsyncInferenceClient(model=config.url, token=config.api_token)
        endpoint_info = await self.client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]


class InferenceAPIAdapter(_HfAdapter):
    async def initialize(self, config: InferenceAPIImplConfig) -> None:
        self.client = AsyncInferenceClient(
            model=config.model_id, token=config.api_token
        )
        endpoint_info = await self.client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]


class InferenceEndpointAdapter(_HfAdapter):
    async def initialize(self, config: InferenceEndpointImplConfig) -> None:
        # Get the inference endpoint details
        api = HfApi(token=config.api_token)
        endpoint = api.get_inference_endpoint(config.endpoint_name)

        # Wait for the endpoint to be ready (if not already)
        endpoint.wait(timeout=60)

        # Initialize the adapter
        self.client = endpoint.async_client
        self.model_id = endpoint.repository
        self.max_tokens = int(
            endpoint.raw["model"]["image"]["custom"]["env"]["MAX_TOTAL_TOKENS"]
        )
