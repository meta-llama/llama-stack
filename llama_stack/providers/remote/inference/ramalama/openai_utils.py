# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    GrammarResponseFormat,
    JsonSchemaResponseFormat,
    Message,
    ToolChoice,
    UserMessage,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
)


def _merge_context_into_content(message: Message) -> Message:  # type: ignore
    """
    Merge the ``context`` field of a Llama Stack ``Message`` object into
    the content field for compabilitiy with OpenAI-style APIs.

    Generates a content string that emulates the current behavior
    of ``llama_models.llama3.api.chat_format.encode_message()``.

    :param message: Message that may include ``context`` field

    :returns: A version of ``message`` with any context merged into the
     ``content`` field.
    """
    if not isinstance(message, UserMessage):  # Separate type check for linter
        return message
    if message.context is None:
        return message
    return UserMessage(
        role=message.role,
        # Emumate llama_models.llama3.api.chat_format.encode_message()
        content=message.content + "\n\n" + message.context,
        context=None,
    )


async def llama_stack_chat_completion_to_openai_chat_completion_dict(
    request: ChatCompletionRequest,
) -> dict:
    """
    Convert a chat completion request in Llama Stack format into an
    equivalent set of arguments to pass to an OpenAI-compatible
    chat completions API.

    :param request: Bundled request parameters in Llama Stack format.

    :returns: Dictionary of key-value pairs to use as an initializer
     for a dataclass or to be converted directly to JSON and sent
     over the wire.
    """

    converted_messages = [
        # This mystery async call makes the parent function also be async
        await convert_message_to_openai_dict(_merge_context_into_content(m), download=True)
        for m in request.messages
    ]
    # converted_tools = _llama_stack_tools_to_openai_tools(request.tools)

    # Llama will try to use built-in tools with no tool catalog, so don't enable
    # tool choice unless at least one tool is enabled.
    converted_tool_choice = "none"
    if (
        request.tool_config is not None
        and request.tool_config.tool_choice == ToolChoice.auto
        and request.tools is not None
        and len(request.tools) > 0
    ):
        converted_tool_choice = "auto"

    # TODO: Figure out what to do with the tool_prompt_format argument.
    #  Other connectors appear to drop it quietly.

    # Use Llama Stack shared code to translate sampling parameters.
    sampling_options = get_sampling_options(request.sampling_params)

    # get_sampling_options() translates repetition penalties to an option that
    # OpenAI's APIs don't know about.
    # vLLM's OpenAI-compatible API also handles repetition penalties wrong.
    # For now, translate repetition penalties into a format that vLLM's broken
    # API will handle correctly. Two wrongs make a right...
    if "repeat_penalty" in sampling_options:
        del sampling_options["repeat_penalty"]
    if request.sampling_params.repetition_penalty is not None and request.sampling_params.repetition_penalty != 1.0:
        sampling_options["repetition_penalty"] = request.sampling_params.repetition_penalty

    # Convert a single response format into four different parameters, per
    # the OpenAI spec
    guided_decoding_options = dict()
    if request.response_format is None:
        # Use defaults
        pass
    elif isinstance(request.response_format, JsonSchemaResponseFormat):
        guided_decoding_options["guided_json"] = request.response_format.json_schema
    elif isinstance(request.response_format, GrammarResponseFormat):
        guided_decoding_options["guided_grammar"] = request.response_format.bnf
    else:
        raise TypeError(f"ResponseFormat object is of unexpected subtype '{type(request.response_format)}'")

    logprob_options = dict()
    if request.logprobs is not None:
        logprob_options["logprobs"] = request.logprobs.top_k

    # Marshall together all the arguments for a ChatCompletionRequest
    request_options = {
        "model": request.model,
        "messages": converted_messages,
        "tool_choice": converted_tool_choice,
        "stream": request.stream,
        **sampling_options,
        **guided_decoding_options,
        **logprob_options,
    }

    return request_options


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import Any, AsyncGenerator, Dict

from openai import AsyncStream
from openai.types.chat.chat_completion import (
    Choice as OpenAIChoice,
)
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import Logprobs as OpenAICompletionLogprobs

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    TokenLogProbs,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.openai_compat import (
    _convert_openai_finish_reason,
    convert_message_to_openai_dict_new,
    convert_tooldef_to_openai_tool,
)


async def convert_chat_completion_request(
    request: ChatCompletionRequest,
    n: int = 1,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    # model -> model
    # messages -> messages
    # sampling_params  TODO(mattf): review strategy
    #  strategy=greedy -> nvext.top_k = -1, temperature = temperature
    #  strategy=top_p -> nvext.top_k = -1, top_p = top_p
    #  strategy=top_k -> nvext.top_k = top_k
    #  temperature -> temperature
    #  top_p -> top_p
    #  top_k -> nvext.top_k
    #  max_tokens -> max_tokens
    #  repetition_penalty -> nvext.repetition_penalty
    # response_format -> GrammarResponseFormat TODO(mf)
    # response_format -> JsonSchemaResponseFormat: response_format = "json_object" & nvext["guided_json"] = json_schema
    # tools -> tools
    # tool_choice ("auto", "required") -> tool_choice
    # tool_prompt_format -> TBD
    # stream -> stream
    # logprobs -> logprobs

    if request.response_format and not isinstance(request.response_format, JsonSchemaResponseFormat):
        raise ValueError(
            f"Unsupported response format: {request.response_format}. Only JsonSchemaResponseFormat is supported."
        )

    nvext = {}
    payload: Dict[str, Any] = dict(
        model=request.model,
        messages=[await convert_message_to_openai_dict_new(message) for message in request.messages],
        stream=request.stream,
        n=n,
        extra_body=dict(nvext=nvext),
        extra_headers={
            b"User-Agent": b"llama-stack: nvidia-inference-adapter",
        },
    )

    if request.response_format:
        # server bug - setting guided_json changes the behavior of response_format resulting in an error
        # payload.update(response_format="json_object")
        nvext.update(guided_json=request.response_format.json_schema)

    if request.tools:
        payload.update(tools=[convert_tooldef_to_openai_tool(tool) for tool in request.tools])
        if request.tool_config.tool_choice:
            payload.update(
                tool_choice=request.tool_config.tool_choice.value
            )  # we cannot include tool_choice w/o tools, server will complain

    if request.logprobs:
        payload.update(logprobs=True)
        payload.update(top_logprobs=request.logprobs.top_k)

    if request.sampling_params:
        nvext.update(repetition_penalty=request.sampling_params.repetition_penalty)

        if request.sampling_params.max_tokens:
            payload.update(max_tokens=request.sampling_params.max_tokens)

        strategy = request.sampling_params.strategy
        if isinstance(strategy, TopPSamplingStrategy):
            nvext.update(top_k=-1)
            payload.update(top_p=strategy.top_p)
            payload.update(temperature=strategy.temperature)
        elif isinstance(strategy, TopKSamplingStrategy):
            if strategy.top_k != -1 and strategy.top_k < 1:
                warnings.warn("top_k must be -1 or >= 1", stacklevel=2)
            nvext.update(top_k=strategy.top_k)
        elif isinstance(strategy, GreedySamplingStrategy):
            nvext.update(top_k=-1)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

    return payload


def convert_completion_request(
    request: CompletionRequest,
    n: int = 1,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    # model -> model
    # prompt -> prompt
    # sampling_params  TODO(mattf): review strategy
    #  strategy=greedy -> nvext.top_k = -1, temperature = temperature
    #  strategy=top_p -> nvext.top_k = -1, top_p = top_p
    #  strategy=top_k -> nvext.top_k = top_k
    #  temperature -> temperature
    #  top_p -> top_p
    #  top_k -> nvext.top_k
    #  max_tokens -> max_tokens
    #  repetition_penalty -> nvext.repetition_penalty
    # response_format -> nvext.guided_json
    # stream -> stream
    # logprobs.top_k -> logprobs

    nvext = {}
    payload: Dict[str, Any] = dict(
        model=request.model,
        prompt=request.content,
        stream=request.stream,
        extra_body=dict(nvext=nvext),
        extra_headers={
            b"User-Agent": b"llama-stack: nvidia-inference-adapter",
        },
        n=n,
    )

    if request.response_format:
        # this is not openai compliant, it is a nim extension
        nvext.update(guided_json=request.response_format.json_schema)

    if request.logprobs:
        payload.update(logprobs=request.logprobs.top_k)

    if request.sampling_params:
        nvext.update(repetition_penalty=request.sampling_params.repetition_penalty)

        if request.sampling_params.max_tokens:
            payload.update(max_tokens=request.sampling_params.max_tokens)

        if request.sampling_params.strategy == "top_p":
            nvext.update(top_k=-1)
            payload.update(top_p=request.sampling_params.top_p)
        elif request.sampling_params.strategy == "top_k":
            if request.sampling_params.top_k != -1 and request.sampling_params.top_k < 1:
                warnings.warn("top_k must be -1 or >= 1", stacklevel=2)
            nvext.update(top_k=request.sampling_params.top_k)
        elif request.sampling_params.strategy == "greedy":
            nvext.update(top_k=-1)
            payload.update(temperature=request.sampling_params.temperature)

    return payload


def _convert_openai_completion_logprobs(
    logprobs: Optional[OpenAICompletionLogprobs],
) -> Optional[List[TokenLogProbs]]:
    """
    Convert an OpenAI CompletionLogprobs into a list of TokenLogProbs.
    """
    if not logprobs:
        return None

    return [TokenLogProbs(logprobs_by_token=logprobs) for logprobs in logprobs.top_logprobs]


def convert_openai_completion_choice(
    choice: OpenAIChoice,
) -> CompletionResponse:
    """
    Convert an OpenAI Completion Choice into a CompletionResponse.
    """
    return CompletionResponse(
        content=choice.text,
        stop_reason=_convert_openai_finish_reason(choice.finish_reason),
        logprobs=_convert_openai_completion_logprobs(choice.logprobs),
    )


async def convert_openai_completion_stream(
    stream: AsyncStream[OpenAICompletion],
) -> AsyncGenerator[CompletionResponse, None]:
    """
    Convert a stream of OpenAI Completions into a stream
    of ChatCompletionResponseStreamChunks.
    """
    async for chunk in stream:
        choice = chunk.choices[0]
        yield CompletionResponseStreamChunk(
            delta=choice.text,
            stop_reason=_convert_openai_finish_reason(choice.finish_reason),
            logprobs=_convert_openai_completion_logprobs(choice.logprobs),
        )
