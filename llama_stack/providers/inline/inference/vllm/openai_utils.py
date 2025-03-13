# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

import vllm

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    GrammarResponseFormat,
    JsonSchemaResponseFormat,
    Message,
    ToolChoice,
    UserMessage,
)
from llama_stack.models.llama.datatypes import BuiltinTool, ToolDefinition
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
)

###############################################################################
# This file contains OpenAI compatibility code that is currently only used
# by the inline vLLM connector. Some or all of this code may be moved to a
# central location at a later date.


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


def _llama_stack_tools_to_openai_tools(
    tools: Optional[List[ToolDefinition]] = None,
) -> List[vllm.entrypoints.openai.protocol.ChatCompletionToolsParam]:
    """
    Convert the list of available tools from Llama Stack's format to vLLM's
    version of OpenAI's format.
    """
    if tools is None:
        return []

    result = []
    for t in tools:
        if isinstance(t.tool_name, BuiltinTool):
            raise NotImplementedError("Built-in tools not yet implemented")
        if t.parameters is None:
            parameters = None
        else:  # if t.parameters is not None
            # Convert the "required" flags to a list of required params
            required_params = [k for k, v in t.parameters.items() if v.required]
            parameters = {
                "type": "object",  # Mystery value that shows up in OpenAI docs
                "properties": {
                    k: {"type": v.param_type, "description": v.description} for k, v in t.parameters.items()
                },
                "required": required_params,
            }

        function_def = vllm.entrypoints.openai.protocol.FunctionDefinition(
            name=t.tool_name, description=t.description, parameters=parameters
        )

        # Every tool definition is double-boxed in a ChatCompletionToolsParam
        result.append(vllm.entrypoints.openai.protocol.ChatCompletionToolsParam(function=function_def))
    return result


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
    converted_tools = _llama_stack_tools_to_openai_tools(request.tools)

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
        "tools": converted_tools,
        "tool_choice": converted_tool_choice,
        "stream": request.stream,
        **sampling_options,
        **guided_decoding_options,
        **logprob_options,
    }

    return request_options
