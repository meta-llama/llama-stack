# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from termcolor import cprint
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_models.datatypes import ModelFamily
from llama_models.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)
from llama_models.sku_list import resolve_model

from llama_stack.providers.utils.inference import supported_inference_models


def augment_messages_for_tools(request: ChatCompletionRequest) -> List[Message]:
    """Reads chat completion request and augments the messages to handle tools.
    For eg. for llama_3_1, add system message with the appropriate tools or
    add user messsage for custom tools, etc.
    """
    model = resolve_model(request.model)
    if model is None:
        cprint(f"Could not resolve model {request.model}", color="red")
        return request.messages

    if model.descriptor() not in supported_inference_models():
        cprint(f"Unsupported inference model? {model.descriptor()}", color="red")
        return request.messages

    if model.model_family == ModelFamily.llama3_1 or (
        model.model_family == ModelFamily.llama3_2
        and is_multimodal(model.core_model_id)
    ):
        # llama3.1 and llama3.2 multimodal models follow the same tool prompt format
        return augment_messages_for_tools_llama_3_1(request)
    elif model.model_family == ModelFamily.llama3_2:
        return augment_messages_for_tools_llama_3_2(request)
    else:
        return request.messages


def augment_messages_for_tools_llama_3_1(
    request: ChatCompletionRequest,
) -> List[Message]:

    assert request.tool_choice == ToolChoice.auto, "Only `ToolChoice.auto` supported"

    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert (
        existing_messages[0].role != Role.system.value
    ), "Should only have 1 system message"

    messages = []

    default_gen = SystemDefaultGenerator()
    default_template = default_gen.gen()

    sys_content = ""

    tool_template = None
    if request.tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(request.tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    sys_content += default_template.render()

    if existing_system_message:
        # TODO: this fn is needed in many places
        def _process(c):
            if isinstance(c, str):
                return c
            else:
                return "<media>"

        sys_content += "\n"

        if isinstance(existing_system_message.content, str):
            sys_content += _process(existing_system_message.content)
        elif isinstance(existing_system_message.content, list):
            sys_content += "\n".join(
                [_process(c) for c in existing_system_message.content]
            )

    messages.append(SystemMessage(content=sys_content))

    has_custom_tools = any(isinstance(dfn.tool_name, str) for dfn in request.tools)
    if has_custom_tools:
        if request.tool_prompt_format == ToolPromptFormat.json:
            tool_gen = JsonCustomToolGenerator()
        elif request.tool_prompt_format == ToolPromptFormat.function_tag:
            tool_gen = FunctionTagCustomToolGenerator()
        else:
            raise ValueError(
                f"Non supported ToolPromptFormat {request.tool_prompt_format}"
            )

        custom_tools = [t for t in request.tools if isinstance(t.tool_name, str)]
        custom_template = tool_gen.gen(custom_tools)
        messages.append(UserMessage(content=custom_template.render()))

    # Add back existing messages from the request
    messages += existing_messages

    return messages


def augment_messages_for_tools_llama_3_2(
    request: ChatCompletionRequest,
) -> List[Message]:
    assert request.tool_choice == ToolChoice.auto, "Only `ToolChoice.auto` supported"

    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert (
        existing_messages[0].role != Role.system.value
    ), "Should only have 1 system message"

    messages = []
    sys_content = ""
    custom_tools, builtin_tools = [], []
    for t in request.tools:
        if isinstance(t.tool_name, str):
            custom_tools.append(t)
        else:
            builtin_tools.append(t)

    tool_template = None
    if builtin_tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(builtin_tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    custom_tools = [dfn for dfn in request.tools if isinstance(dfn.tool_name, str)]
    if custom_tools:
        if request.tool_prompt_format != ToolPromptFormat.python_list:
            raise ValueError(
                f"Non supported ToolPromptFormat {request.tool_prompt_format}"
            )

        tool_gen = PythonListCustomToolGenerator()
        tool_template = tool_gen.gen(custom_tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    if existing_system_message:
        sys_content += interleaved_text_media_as_str(
            existing_system_message.content, sep="\n"
        )

    messages.append(SystemMessage(content=sys_content))

    # Add back existing messages from the request
    messages += existing_messages
    return messages
