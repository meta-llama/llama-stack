# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from typing import List

from llama_toolchain.inference.api import (
    BuiltinTool,
    Message,
    SystemMessage,
    ToolDefinition,
)

from .tools.builtin import SingleMessageBuiltinTool


def get_agentic_prefix_messages(
    builtin_tools: List[SingleMessageBuiltinTool], custom_tools: List[ToolDefinition]
) -> List[Message]:
    messages = []
    content = ""
    if builtin_tools:
        content += "Environment: ipython\n"

        tool_str = ", ".join(
            [
                t.get_name()
                for t in builtin_tools
                if t.get_name() != BuiltinTool.code_interpreter.value
            ]
        )
        if tool_str:
            content += f"Tools: {tool_str}\n"

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")
    date_str = f"""
Cutting Knowledge Date: December 2023
Today Date: {formatted_date}\n\n"""
    content += date_str

    if custom_tools:
        custom_message = get_system_prompt_for_custom_tools(custom_tools)
        content += custom_message

    # TODO: Replace this hard coded message with instructions coming in the request
    if False:
        content += "You are a helpful Assistant."

    messages.append(SystemMessage(content=content))
    return messages


def get_system_prompt_for_custom_tools(custom_tools: List[ToolDefinition]) -> str:
    custom_tool_params = ""
    for t in custom_tools:
        custom_tool_params += get_instruction_string(t) + "\n"
        custom_tool_params += get_parameters_string(t) + "\n\n"

    content = f"""
You have access to the following functions:

{custom_tool_params}
Think very carefully before calling functions.
If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

"""
    return content


def get_instruction_string(custom_tool_definition) -> str:
    return f"Use the function '{custom_tool_definition.tool_name}' to '{custom_tool_definition.description}'"


def get_parameters_string(custom_tool_definition) -> str:
    return json.dumps(
        {
            "name": custom_tool_definition.tool_name,
            "description": custom_tool_definition.description,
            "parameters": {
                name: definition.__dict__
                for name, definition in custom_tool_definition.parameters.items()
            },
        }
    )


# NOTE: Unused right now
def translate_custom_tool_definition_to_json(tool_def):
    """Translates ToolDefinition to json as expected by model
    eg. output for a function
    {
        "type": "function",
        "function": {
            "name": "conv_int",
            "description": "Convert serialized fract24 integer into int value.",
            "parameters": {
                "type": "object",
                "properties": [
                    {
                        "data": {
                            "type": "object",
                            "description": ""
                        }
                    }
                ],
                "required": ["data"]
            }
        }
    }
    """
    assert isinstance(tool_def.tool_name, str)
    func_def = {"type": "function", "function": {}}
    func_def["function"]["name"] = tool_def.tool_name
    func_def["function"]["description"] = tool_def.description or ""
    if tool_def.parameters:
        required = []
        properties = []
        for p_name, p_def in tool_def.parameters.items():
            properties.append(
                {
                    p_name: {
                        # TODO: see if this should not always be object
                        "type": "object",
                        "description": p_def.description or "",
                    }
                }
            )
            if p_def.required:
                required.append(p_name)
        func_def["function"]["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    else:
        func_def["function"]["parameters"] = {}

    return json.dumps(func_def)
