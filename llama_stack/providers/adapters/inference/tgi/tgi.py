# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any, AsyncGenerator, Dict

import requests

from huggingface_hub import HfApi, InferenceClient
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.prepare_messages import prepare_messages

from .config import TGIImplConfig

HF_SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


class TGIAdapter(Inference):
    def __init__(self, config: TGIImplConfig) -> None:
        self.config = config
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)

    @property
    def client(self) -> InferenceClient:
        return InferenceClient(model=self.config.url, token=self.config.api_token)

    def _get_endpoint_info(self) -> Dict[str, Any]:
        return {
            **self.client.get_endpoint_info(),
            "inference_url": self.config.url,
        }

    async def initialize(self) -> None:
        try:
            info = self._get_endpoint_info()
            if "model_id" not in info:
                raise RuntimeError("Missing model_id in model info")
            if "max_total_tokens" not in info:
                raise RuntimeError("Missing max_total_tokens in model info")
            self.max_tokens = info["max_total_tokens"]

            model_id = info["model_id"]
            model_name = next(
                (name for name, id in HF_SUPPORTED_MODELS.items() if id == model_id),
                None,
            )
            if model_name is None:
                raise RuntimeError(
                    f"TGI is serving model: {model_id}, use one of the supported models: {', '.join(HF_SUPPORTED_MODELS.values())}"
                )
            self.model_name = model_name
            self.inference_url = info["inference_url"]
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Error initializing TGIAdapter: {e}") from e

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
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

        messages = prepare_messages(request)
        model_input = self.formatter.encode_dialog_prompt(messages)
        prompt = self.tokenizer.decode(model_input.tokens)

        input_tokens = len(model_input.tokens)
        max_new_tokens = min(
            request.sampling_params.max_tokens or (self.max_tokens - input_tokens),
            self.max_tokens - input_tokens - 1,
        )

        print(f"Calculated max_new_tokens: {max_new_tokens}")

        assert (
            request.model == self.model_name
        ), f"Model mismatch, expected {self.model_name}, got {request.model}"

        options = self.get_chat_options(request)
        if not request.stream:
            response = self.client.text_generation(
                prompt=prompt,
                stream=False,
                details=True,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            )
            stop_reason = None
            if response.details.finish_reason:
                if response.details.finish_reason == "stop":
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

            for response in self.client.text_generation(
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


class InferenceEndpointAdapter(TGIAdapter):
    def __init__(self, config: TGIImplConfig) -> None:
        super().__init__(config)
        self.config.url = self._construct_endpoint_url()

    def _construct_endpoint_url(self) -> str:
        hf_endpoint_name = self.config.hf_endpoint_name
        assert hf_endpoint_name.count("/") <= 1, (
            "Endpoint name must be in the format of 'namespace/endpoint_name' "
            "or 'endpoint_name'"
        )
        if "/" not in hf_endpoint_name:
            hf_namespace: str = self.get_namespace()
            endpoint_path = f"{hf_namespace}/{hf_endpoint_name}"
        else:
            endpoint_path = hf_endpoint_name
        return f"https://api.endpoints.huggingface.cloud/v2/endpoint/{endpoint_path}"

    def get_namespace(self) -> str:
        return HfApi().whoami()["name"]

    @property
    def client(self) -> InferenceClient:
        return InferenceClient(model=self.inference_url, token=self.config.api_token)

    def _get_endpoint_info(self) -> Dict[str, Any]:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.config.api_token}",
        }
        response = requests.get(self.config.url, headers=headers)
        response.raise_for_status()
        endpoint_info = response.json()
        return {
            "inference_url": endpoint_info["status"]["url"],
            "model_id": endpoint_info["model"]["repository"],
            "max_total_tokens": int(
                endpoint_info["model"]["image"]["custom"]["env"]["MAX_TOTAL_TOKENS"]
            ),
        }

    async def initialize(self) -> None:
        await super().initialize()
