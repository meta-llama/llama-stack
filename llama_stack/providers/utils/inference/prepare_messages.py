# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_models.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    SystemDefaultGenerator,
)


def prepare_messages(request: ChatCompletionRequest) -> List[Message]:

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
