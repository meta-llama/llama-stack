# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
import uuid
from typing import AsyncGenerator, Optional

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.sku_list import resolve_model

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VLLMSamplingParams

from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate
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


class VLLMInferenceImpl(Inference, ModelsProtocolPrivate):
    """Inference implementation for vLLM."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine = None
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self):
        log.info("Initializing vLLM inference provider.")

        # Disable usage stats reporting. This would be a surprising thing for most
        # people to find out was on by default.
        # https://docs.vllm.ai/en/latest/serving/usage_stats.html
        if "VLLM_NO_USAGE_STATS" not in os.environ:
            os.environ["VLLM_NO_USAGE_STATS"] = "1"

        model = resolve_model(self.config.model)
        if model is None:
            raise ValueError(f"Unknown model {self.config.model}")

        if model.huggingface_repo is None:
            raise ValueError(f"Model {self.config.model} needs a huggingface repo")

        # TODO -- there are a ton of options supported here ...
        engine_args = AsyncEngineArgs(
            model=model.huggingface_repo,
            tokenizer=model.huggingface_repo,
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=self.config.enforce_eager,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            guided_decoding_backend="lm-format-enforcer",
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def shutdown(self):
        """Shut down the vLLM inference adapter."""
        log.info("Shutting down vLLM inference provider.")
        if self.engine:
            self.engine.shutdown_background_loop()

    # Note that the return type of the superclass method is WRONG
    async def register_model(self, model: Model) -> Model:
        """
        Callback that is called when the server associates an inference endpoint
        with an inference provider.

        :param model: Object that encapsulates parameters necessary for identifying
         a specific LLM.

        :returns: The input ``Model`` object. It may or may not be permissible
         to change fields before returning this object.
        """
        log.info(f"Registering model {model.identifier} with vLLM inference provider.")
        # The current version of this provided is hard-coded to serve only
        # the model specified in the YAML config file.
        configured_model = resolve_model(self.config.model)
        registered_model = resolve_model(model.model_id)

        if configured_model.core_model_id != registered_model.core_model_id:
            raise ValueError(
                f"Requested model '{model.identifier}' is different from "
                f"model '{self.config.model}' that this provider "
                f"is configured to serve"
            )
        return model

    def _sampling_params(self, sampling_params: SamplingParams) -> VLLMSamplingParams:
        if sampling_params is None:
            return VLLMSamplingParams(max_tokens=self.config.max_tokens)

        # TODO convert what I saw in my first test ... but surely there's more to do here
        kwargs = {
            "temperature": sampling_params.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if sampling_params.top_k:
            kwargs["top_k"] = sampling_params.top_k
        if sampling_params.top_p:
            kwargs["top_p"] = sampling_params.top_p
        if sampling_params.max_tokens:
            kwargs["max_tokens"] = sampling_params.max_tokens
        if sampling_params.repetition_penalty > 0:
            kwargs["repetition_penalty"] = sampling_params.repetition_penalty

        return VLLMSamplingParams(**kwargs)

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> CompletionResponse | CompletionResponseStreamChunk:
        raise NotImplementedError("Completion not implemented for vLLM")

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> ChatCompletionResponse | ChatCompletionResponseStreamChunk:
        assert self.engine is not None

        request = ChatCompletionRequest(
            model=model_id,
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

        prompt = await chat_completion_request_to_prompt(
            request, self.config.model, self.formatter
        )
        vllm_sampling_params = self._sampling_params(request.sampling_params)
        results_generator = self.engine.generate(
            prompt, vllm_sampling_params, request_id
        )
        if stream:
            return self._stream_chat_completion(request, results_generator)
        else:
            return await self._nonstream_chat_completion(request, results_generator)

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
            cur = []
            async for chunk in results_generator:
                if not chunk.outputs:
                    log.warning("Empty chunk received")
                    continue

                output = chunk.outputs[-1]

                new_tokens = output.token_ids[len(cur) :]
                text = self.formatter.tokenizer.decode(new_tokens)
                cur.extend(new_tokens)
                choice = OpenAICompatCompletionChoice(
                    finish_reason=output.finish_reason,
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
        self, model_id: str, contents: List[InterleavedContent]
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
