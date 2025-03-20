# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re
import uuid
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

# These vLLM modules contain names that overlap with Llama Stack names, so we import
# fully-qualified names
import vllm.entrypoints.openai.protocol
import vllm.sampling_params
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextDelta,
    ToolCallDelta,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    GrammarResponseFormat,
    Inference,
    JsonSchemaResponseFormat,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    TokenLogProbs,
    ToolChoice,
    ToolConfig,
)
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.models.llama import sku_list
from llama_stack.models.llama.datatypes import (
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.providers.remote.inference.vllm.vllm import build_hf_repo_model_entries
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    ModelsProtocolPrivate,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    get_stop_reason,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .config import VLLMConfig
from .openai_utils import llama_stack_chat_completion_to_openai_chat_completion_dict

# Map from Hugging Face model architecture name to appropriate tool parser.
# See vllm.entrypoints.openai.tool_parsers.ToolParserManager.tool_parsers for the full list of
# available parsers.
# TODO: Expand this list
CONFIG_TYPE_TO_TOOL_PARSER = {
    "GraniteConfig": "granite",
    "MllamaConfig": "llama3_json",
    "LlamaConfig": "llama3_json",
}
DEFAULT_TOOL_PARSER = "pythonic"


logger = get_logger(__name__, category="inference")


def _random_uuid_str() -> str:
    return str(uuid.uuid4().hex)


def _response_format_to_guided_decoding_params(
    response_format: Optional[ResponseFormat],  # type: ignore
) -> vllm.sampling_params.GuidedDecodingParams:
    """
    Translate constrained decoding parameters from Llama Stack's format to vLLM's format.

    :param response_format: Llama Stack version of constrained decoding info. Can be ``None``,
     indicating no constraints.
    :returns: The equivalent dataclass object for the low-level inference layer of vLLM.
    """
    if response_format is None:
        # As of vLLM 0.6.3, the default constructor for GuidedDecodingParams() returns an invalid
        # value that crashes the executor on some code paths. Use ``None`` instead.
        return None

    # Llama Stack currently implements fewer types of constrained decoding than vLLM does.
    # Translate the types that exist and detect if Llama Stack adds new ones.
    if isinstance(response_format, JsonSchemaResponseFormat):
        return vllm.sampling_params.GuidedDecodingParams(json=response_format.json_schema)
    elif isinstance(response_format, GrammarResponseFormat):
        # BNF grammar.
        # Llama Stack uses the parse tree of the grammar, while vLLM uses the string
        # representation of the grammar.
        raise TypeError(
            "Constrained decoding with BNF grammars is not currently implemented, because the "
            "reference implementation does not implement it."
        )
    else:
        raise TypeError(f"ResponseFormat object is of unexpected subtype '{type(response_format)}'")


def _convert_sampling_params(
    sampling_params: Optional[SamplingParams],
    response_format: Optional[ResponseFormat],  # type: ignore
    log_prob_config: Optional[LogProbConfig],
) -> vllm.SamplingParams:
    """Convert sampling and constrained decoding configuration from Llama Stack's format to vLLM's
    format."""
    # In the absence of provided config values, use Llama Stack defaults as encoded in the Llama
    # Stack dataclasses. These defaults are different from vLLM's defaults.
    if sampling_params is None:
        sampling_params = SamplingParams()
    if log_prob_config is None:
        log_prob_config = LogProbConfig()

    if isinstance(sampling_params.strategy, TopKSamplingStrategy):
        if sampling_params.strategy.top_k == 0:
            # vLLM treats "k" differently for top-k sampling
            vllm_top_k = -1
        else:
            vllm_top_k = sampling_params.strategy.top_k
    else:
        vllm_top_k = -1

    if isinstance(sampling_params.strategy, TopPSamplingStrategy):
        vllm_top_p = sampling_params.strategy.top_p
        # Llama Stack only allows temperature with top-P.
        vllm_temperature = sampling_params.strategy.temperature
    else:
        vllm_top_p = 1.0
        vllm_temperature = 0.0

    # vLLM allows top-p and top-k at the same time.
    vllm_sampling_params = vllm.SamplingParams.from_optional(
        max_tokens=(None if sampling_params.max_tokens == 0 else sampling_params.max_tokens),
        temperature=vllm_temperature,
        top_p=vllm_top_p,
        top_k=vllm_top_k,
        repetition_penalty=sampling_params.repetition_penalty,
        guided_decoding=_response_format_to_guided_decoding_params(response_format),
        logprobs=log_prob_config.top_k,
    )
    return vllm_sampling_params


class VLLMInferenceImpl(Inference, ModelsProtocolPrivate):
    """
    vLLM-based inference model adapter for Llama Stack with support for multiple models.

    Requires the configuration parameters documented in the :class:`VllmConfig2` class.
    """

    config: VLLMConfig
    register_helper: ModelRegistryHelper
    model_ids: set[str]
    resolved_model_id: str | None
    engine: AsyncLLMEngine | None
    chat: OpenAIServingChat | None
    is_meta_llama_model: bool

    def __init__(self, config: VLLMConfig):
        self.config = config
        logger.info(f"Config is: {self.config}")

        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.formatter = ChatFormat(Tokenizer.get_instance())

        # The following are initialized when paths are bound to this provider
        self.resolved_model_id = None
        self.model_ids = set()
        self.engine = None
        self.chat = None
        self.is_meta_llama_model = False

    ###########################################################################
    # METHODS INHERITED FROM IMPLICIT BASE CLASS.
    # TODO: Make this class inherit from the new base class ProviderBase once that class exists.

    async def initialize(self) -> None:
        """
        Callback that is invoked through many levels of indirection during provider class
        instantiation, sometime after when __init__() is called and before any model registration
        methods or methods connected to a REST API are called.

        It's not clear what assumptions the class can make about the platform's initialization
        state here that can't be made during __init__(), and vLLM can't be started until we know
        what model it's supposed to be serving, so nothing happens here currently.
        """
        pass

    async def shutdown(self) -> None:
        logger.info(f"Shutting down inline vLLM inference provider {self}.")
        if self.engine is not None:
            self.engine.shutdown_background_loop()
            self.engine = None
            self.chat = None
            self.model_ids = set()
            self.resolved_model_id = None

    ###########################################################################
    # METHODS INHERITED FROM ModelsProtocolPrivate INTERFACE

    # Note that the return type of the superclass method is WRONG
    async def register_model(self, model: Model) -> Model:
        """
        Callback that is called when the server associates an inference endpoint with an
        inference provider.

        :param model: Object that encapsulates parameters necessary for identifying a specific
         LLM.

        :returns: The input ``Model`` object. It may or may not be permissible to change fields
         before returning this object.
        """
        logger.debug(f"In register_model({model})")

        # First attempt to interpret the model coordinates as a Llama model name
        resolved_llama_model = sku_list.resolve_model(model.provider_model_id)
        if resolved_llama_model is not None:
            # Load from Hugging Face repo into default local cache dir
            model_id_for_vllm = resolved_llama_model.huggingface_repo

            # Detect a genuine Meta Llama model to trigger Meta-specific preprocessing.
            # Don't set self.is_meta_llama_model until we actually load the model.
            is_meta_llama_model = True
        else:  # if resolved_llama_model is None
            # Not a Llama model name. Pass the model id through to vLLM's loader
            model_id_for_vllm = model.provider_model_id
            is_meta_llama_model = False

        if self.resolved_model_id is not None:
            if model_id_for_vllm != self.resolved_model_id:
                raise ValueError(
                    f"Attempted to serve two LLMs (ids '{self.resolved_model_id}') and "
                    f"'{model_id_for_vllm}') from one copy of provider '{self}'. Use multiple "
                    f"copies of the provider instead."
                )
            else:
                # Model already loaded
                logger.info(
                    f"Requested id {model} resolves to {model_id_for_vllm}, which is already loaded. Continuing."
                )
                self.model_ids.add(model.model_id)
                return model

        logger.info(f"Requested id {model} resolves to {model_id_for_vllm}. Loading {model_id_for_vllm}.")
        if is_meta_llama_model:
            logger.info(f"Model {model_id_for_vllm} is a Meta Llama model.")
        self.is_meta_llama_model = is_meta_llama_model

        # If we get here, this is the first time registering a model.
        # Preload so that the first inference request won't time out.
        engine_args = AsyncEngineArgs(
            model=model_id_for_vllm,
            tokenizer=model_id_for_vllm,
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=self.config.enforce_eager,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # vLLM currently requires the user to specify the tool parser manually. To choose a tool
        # parser, we need to determine what model architecture is being used. For now, we infer
        # that information from what config class the model uses.
        low_level_model_config = self.engine.engine.get_model_config()
        hf_config = low_level_model_config.hf_config
        hf_config_class_name = hf_config.__class__.__name__
        if hf_config_class_name in CONFIG_TYPE_TO_TOOL_PARSER:
            tool_parser = CONFIG_TYPE_TO_TOOL_PARSER[hf_config_class_name]
        else:
            # No info -- choose a default so we can at least attempt tool
            # use.
            tool_parser = DEFAULT_TOOL_PARSER
        logger.debug(f"{hf_config_class_name=}")
        logger.debug(f"{tool_parser=}")

        # Wrap the lower-level engine in an OpenAI-compatible chat API
        model_config = await self.engine.get_model_config()
        self.chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=[
                    # The layer below us will only see resolved model IDs
                    BaseModelPath(model_id_for_vllm, model_id_for_vllm)
                ],
            ),
            response_role="assistant",
            request_logger=None,  # Use default logging
            chat_template=None,  # Use default template from model checkpoint
            enable_auto_tools=True,
            tool_parser=tool_parser,
            chat_template_content_format="auto",
        )
        self.resolved_model_id = model_id_for_vllm
        self.model_ids.add(model.model_id)

        logger.info(f"Finished preloading model: {model_id_for_vllm}")

        return model

    async def unregister_model(self, model_id: str) -> None:
        """
        Callback that is called when the server removes an inference endpoint from an inference
        provider.

        :param model_id: The same external ID that the higher layers of the stack previously passed
        to :func:`register_model()`
        """
        if model_id not in self.model_ids:
            raise ValueError(
                f"Attempted to unregister model ID '{model_id}', but that ID is not registered to this provider."
            )
        self.model_ids.remove(model_id)

        if len(self.model_ids) == 0:
            # Last model was just unregistered. Shut down the connection to vLLM and free up
            # resources.
            # Note that this operation may cause in-flight chat completion requests on the
            # now-unregistered model to return errors.
            self.resolved_model_id = None
            self.chat = None
            self.engine.shutdown_background_loop()
            self.engine = None

    ###########################################################################
    # METHODS INHERITED FROM Inference INTERFACE

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        if model_id not in self.model_ids:
            raise ValueError(
                f"This adapter is not registered to model id '{model_id}'. Registered IDs are: {self.model_ids}"
            )
        if not isinstance(content, str):
            raise NotImplementedError("Multimodal input not currently supported")
        if sampling_params is None:
            sampling_params = SamplingParams()

        converted_sampling_params = _convert_sampling_params(sampling_params, response_format, logprobs)

        logger.debug(f"{converted_sampling_params=}")

        if stream:
            return self._streaming_completion(content, converted_sampling_params)
        else:
            streaming_result = None
            async for _ in self._streaming_completion(content, converted_sampling_params):
                pass
            return CompletionResponse(
                content=streaming_result.delta,
                stop_reason=streaming_result.stop_reason,
                logprobs=streaming_result.logprobs,
            )

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],  # type: ignore
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,  # type: ignore
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> ChatCompletionResponse | ChatCompletionResponseStreamChunk:
        sampling_params = sampling_params or SamplingParams()
        if model_id not in self.model_ids:
            raise ValueError(
                f"This adapter is not registered to model id '{model_id}'. Registered IDs are: {self.model_ids}"
            )

        # Convert to Llama Stack internal format for consistency
        request = ChatCompletionRequest(
            model=self.resolved_model_id,
            messages=messages,
            sampling_params=sampling_params,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        if self.is_meta_llama_model:
            # Bypass vLLM chat templating layer for Meta Llama models, because the
            # templating layer in Llama Stack currently produces better results.
            logger.debug(
                f"Routing {self.resolved_model_id} chat completion through "
                f"Llama Stack's templating layer instead of vLLM's."
            )
            return await self._chat_completion_for_meta_llama(request)

        logger.debug(f"{self.resolved_model_id} is not a Meta Llama model")

        # Arguments to the vLLM call must be packaged as a ChatCompletionRequest dataclass.
        # Note that this dataclass has the same name as a similar dataclass in Llama Stack.
        request_options = await llama_stack_chat_completion_to_openai_chat_completion_dict(request)
        chat_completion_request = vllm.entrypoints.openai.protocol.ChatCompletionRequest(**request_options)

        logger.debug(f"Converted request: {chat_completion_request}")

        vllm_result = await self.chat.create_chat_completion(chat_completion_request)
        logger.debug(f"Result from vLLM: {vllm_result}")
        if isinstance(vllm_result, vllm.entrypoints.openai.protocol.ErrorResponse):
            raise ValueError(f"Error from vLLM layer: {vllm_result}")

        # Return type depends on "stream" argument
        if stream:
            if not isinstance(vllm_result, AsyncGenerator):
                raise TypeError(f"Unexpected result type {type(vllm_result)} for streaming inference call")
            # vLLM client returns a stream of strings, which need to be parsed.
            # Stream comes in the form of an async generator.
            return self._convert_streaming_results(vllm_result)
        else:
            if not isinstance(vllm_result, vllm.entrypoints.openai.protocol.ChatCompletionResponse):
                raise TypeError(f"Unexpected result type {type(vllm_result)} for non-streaming inference call")
            return self._convert_non_streaming_results(vllm_result)

    ###########################################################################
    # INTERNAL METHODS

    async def _streaming_completion(
        self, content: str, sampling_params: vllm.SamplingParams
    ) -> AsyncIterator[CompletionResponseStreamChunk]:
        """Internal implementation of :func:`completion()` API for the streaming case. Assumes
        that arguments have been validated upstream.

        :param content: Must be a string
        :param sampling_params: Paramters from  public API's ``response_format``
         and ``sampling_params`` arguments, converted to VLLM format
        """
        # We run agains the vLLM generate() call directly instead of using the OpenAI-compatible
        # layer, because doing so simplifies the code here.

        # The vLLM engine requires a unique identifier for each call to generate()
        request_id = _random_uuid_str()

        # The vLLM generate() API is streaming-only and returns an async generator.
        # The generator returns objects of type vllm.RequestOutput.
        results_generator = self.engine.generate(content, sampling_params, request_id)

        # Need to know the model's EOS token ID for the conversion code below.
        # AsyncLLMEngine is a wrapper around LLMEngine, and the tokenizer is only available if
        # we drill down to the LLMEngine inside the AsyncLLMEngine.
        # Similarly, the tokenizer in an LLMEngine is a wrapper around a BaseTokenizerGroup,
        # and we need to drill down to the Hugging Face tokenizer inside the BaseTokenizerGroup.
        llm_engine = self.engine.engine
        tokenizer_group = llm_engine.tokenizer
        eos_token_id = tokenizer_group.tokenizer.eos_token_id

        request_output: vllm.RequestOutput = None
        async for request_output in results_generator:
            # Check for weird inference failures
            if request_output.outputs is None or len(request_output.outputs) == 0:
                # This case also should never happen
                raise ValueError("Inference produced empty result")

            # If we get here, then request_output contains the final output of the generate() call.
            # The result may include multiple alternate outputs, but Llama Stack APIs only allow
            # us to return one.
            output: vllm.CompletionOutput = request_output.outputs[0]
            completion_string = output.text

            # Convert logprobs from vLLM's format to Llama Stack's format
            logprobs = [
                TokenLogProbs(logprobs_by_token={v.decoded_token: v.logprob for _, v in logprob_dict.items()})
                for logprob_dict in output.logprobs
            ]

            # The final output chunk should be labeled with the reason that the overall generate()
            # call completed.
            logger.debug(f"{output.stop_reason=}; {type(output.stop_reason)=}")
            if output.stop_reason is None:
                stop_reason = None  # Still going
            elif output.stop_reason == "stop":
                stop_reason = StopReason.end_of_turn
            elif output.stop_reason == "length":
                stop_reason = StopReason.out_of_tokens
            elif isinstance(output.stop_reason, int):
                # If the model config specifies multiple end-of-sequence tokens, then vLLM
                # will return the token ID of the EOS token in the stop_reason field.
                stop_reason = StopReason.end_of_turn
            else:
                raise ValueError(f"Unrecognized stop reason '{output.stop_reason}'")

            # vLLM's protocol outputs the stop token, then sets end of message on the next step for
            # some reason.
            if request_output.outputs[-1].token_ids[-1] == eos_token_id:
                stop_reason = StopReason.end_of_message

            yield CompletionResponseStreamChunk(delta=completion_string, stop_reason=stop_reason, logprobs=logprobs)

        # Llama Stack requires that the last chunk have a stop reason, but vLLM doesn't always
        # provide one if it runs out of tokens.
        if stop_reason is None:
            yield CompletionResponseStreamChunk(
                delta=completion_string,
                stop_reason=StopReason.out_of_tokens,
                logprobs=logprobs,
            )

    def _convert_non_streaming_results(
        self, vllm_result: vllm.entrypoints.openai.protocol.ChatCompletionResponse
    ) -> ChatCompletionResponse:
        """
        Subroutine to convert the non-streaming output of vLLM's OpenAI-compatible API into an
        equivalent Llama Stack object.

        The result from vLLM's non-streaming API is a dataclass with the same name as the Llama
        Stack ChatCompletionResponse dataclass, but with more and different field names. We ignore
        the fields that aren't currently present in the Llama Stack dataclass.
        """

        # There may be multiple responses, but we can only pass through the first one.
        if len(vllm_result.choices) == 0:
            raise ValueError("Don't know how to convert response object without any responses")
        vllm_message = vllm_result.choices[0].message
        vllm_finish_reason = vllm_result.choices[0].finish_reason

        converted_message = CompletionMessage(
            role=vllm_message.role,
            # Llama Stack API won't accept None for content field.
            content=("" if vllm_message.content is None else vllm_message.content),
            stop_reason=get_stop_reason(vllm_finish_reason),
            tool_calls=[
                ToolCall(
                    call_id=t.id,
                    tool_name=t.function.name,
                    # vLLM function args come back as a string. Llama Stack expects JSON.
                    arguments=json.loads(t.function.arguments),
                    arguments_json=t.function.arguments,
                )
                for t in vllm_message.tool_calls
            ],
        )

        # TODO: Convert logprobs

        logger.debug(f"Converted message: {converted_message}")

        return ChatCompletionResponse(
            completion_message=converted_message,
        )

    async def _chat_completion_for_meta_llama(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        """
        Subroutine that routes chat completions for Meta Llama models through Llama Stack's
        chat template instead of using vLLM's version of that template. The Llama Stack version
        of the chat template currently produces more reliable outputs.

        Once vLLM's support for Meta Llama models has matured more, we should consider routing
        Meta Llama requests through the vLLM chat completions API instead of using this method.
        """
        formatter = ChatFormat(Tokenizer.get_instance())

        # Note that this function call modifies `request` in place.
        prompt = await chat_completion_request_to_prompt(request, self.resolved_model_id)

        model_id = list(self.model_ids)[0]  # Any model ID will do here
        completion_response_or_iterator = await self.completion(
            model_id=model_id,
            content=prompt,
            sampling_params=request.sampling_params,
            response_format=request.response_format,
            stream=request.stream,
            logprobs=request.logprobs,
        )

        if request.stream:
            if not isinstance(completion_response_or_iterator, AsyncIterator):
                raise TypeError(
                    f"Received unexpected result type {type(completion_response_or_iterator)}for streaming request."
                )
            return self._chat_completion_for_meta_llama_streaming(completion_response_or_iterator, request)

        # elsif not request.stream:
        if not isinstance(completion_response_or_iterator, CompletionResponse):
            raise TypeError(
                f"Received unexpected result type {type(completion_response_or_iterator)}for non-streaming request."
            )
        completion_response: CompletionResponse = completion_response_or_iterator
        raw_message = formatter.decode_assistant_message_from_content(
            completion_response.content, completion_response.stop_reason
        )
        return ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=raw_message.content,
                stop_reason=raw_message.stop_reason,
                tool_calls=raw_message.tool_calls,
            ),
            logprobs=completion_response.logprobs,
        )

    async def _chat_completion_for_meta_llama_streaming(
        self, results_iterator: AsyncIterator, request: ChatCompletionRequest
    ) -> AsyncIterator:
        """
        Code from :func:`_chat_completion_for_meta_llama()` that needs to be a separate
        method to keep asyncio happy.
        """

        # Convert to OpenAI format, then use shared code to convert to Llama Stack format.
        async def _generate_and_convert_to_openai_compat():
            chunk: CompletionResponseStreamChunk  # Make Pylance happy
            last_text_len = 0
            async for chunk in results_iterator:
                if chunk.stop_reason == StopReason.end_of_turn:
                    finish_reason = "stop"
                elif chunk.stop_reason == StopReason.end_of_message:
                    finish_reason = "eos"
                elif chunk.stop_reason == StopReason.out_of_tokens:
                    finish_reason = "length"
                else:
                    finish_reason = None

                # Convert delta back to an actual delta
                text_delta = chunk.delta[last_text_len:]
                last_text_len = len(chunk.delta)

                logger.debug(f"{text_delta=}; {finish_reason=}")

                yield OpenAICompatCompletionResponse(
                    choices=[OpenAICompatCompletionChoice(finish_reason=finish_reason, text=text_delta)]
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(stream, request):
            logger.debug(f"Returning chunk: {chunk}")
            yield chunk

    async def _convert_streaming_results(self, vllm_result: AsyncIterator) -> AsyncIterator:
        """
        Subroutine that wraps the streaming outputs of vLLM's OpenAI-compatible
        API into a second async iterator that returns Llama Stack objects.

        :param vllm_result: Stream of strings that need to be parsed
        """
        # Tool calls come in pieces, but Llama Stack expects them in bigger chunks. We build up
        # those chunks and output them at the end.
        # This data structure holds the current set of partial tool calls.
        index_to_tool_call: Dict[int, Dict] = dict()

        # The Llama Stack event stream must always start with a start event. Use an empty one to
        # simplify logic below
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta=TextDelta(text=""),
                stop_reason=None,
            )
        )

        converted_stop_reason = None
        async for chunk_str in vllm_result:
            # Due to OpenAI compatibility, each event in the stream will start with "data: " and
            # end with "\n\n".
            _prefix = "data: "
            _suffix = "\n\n"
            if not chunk_str.startswith(_prefix) or not chunk_str.endswith(_suffix):
                raise ValueError(f"Can't parse result string from vLLM: '{re.escape(chunk_str)}'")

            # In between the "data: " and newlines is an event record
            data_str = chunk_str[len(_prefix) : -len(_suffix)]

            # The end of the stream is indicated with "[DONE]"
            if data_str == "[DONE]":
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.complete,
                        delta=TextDelta(text=""),
                        stop_reason=converted_stop_reason,
                    )
                )
                return

            # Anything that is not "[DONE]" should be a JSON record
            parsed_chunk = json.loads(data_str)

            logger.debug(f"Parsed JSON event to:\n{json.dumps(parsed_chunk, indent=2)}")

            # The result may contain multiple completions, but Llama Stack APIs only support
            # returning one.
            first_choice = parsed_chunk["choices"][0]
            converted_stop_reason = get_stop_reason(first_choice["finish_reason"])
            delta_record = first_choice["delta"]

            if "content" in delta_record:
                # Text delta
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=TextDelta(text=delta_record["content"]),
                        stop_reason=converted_stop_reason,
                    )
                )
            elif "tool_calls" in delta_record:
                # Tool call(s). Llama Stack APIs do not have a clear way to return partial tool
                # calls, so buffer until we get a "tool calls" stop reason
                for tc in delta_record["tool_calls"]:
                    index = tc["index"]
                    if index not in index_to_tool_call:
                        # First time this tool call is showing up
                        index_to_tool_call[index] = dict()
                    tool_call = index_to_tool_call[index]
                    if "id" in tc:
                        tool_call["call_id"] = tc["id"]
                    if "function" in tc:
                        if "name" in tc["function"]:
                            tool_call["tool_name"] = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            # Arguments comes in as pieces of a string
                            if "arguments_str" not in tool_call:
                                tool_call["arguments_str"] = ""
                            tool_call["arguments_str"] += tc["function"]["arguments"]
            else:
                raise ValueError(f"Don't know how to parse event delta: {delta_record}")

            if first_choice["finish_reason"] == "tool_calls":
                # Special OpenAI code for "tool calls complete".
                # Output the buffered tool calls. Llama Stack requires a separate event per tool
                # call.
                for tool_call_record in index_to_tool_call.values():
                    # Arguments come in as a string. Parse the completed string.
                    tool_call_record["arguments"] = json.loads(tool_call_record["arguments_str"])
                    del tool_call_record["arguments_str"]

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(tool_call=tool_call_record, parse_status="succeeded"),
                            stop_reason=converted_stop_reason,
                        )
                    )

        # If we get here, we've lost the connection with the vLLM event stream before it ended
        # normally.
        raise ValueError("vLLM event stream ended without [DONE] message.")
