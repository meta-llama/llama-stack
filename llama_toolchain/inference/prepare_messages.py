import json
import os
import textwrap

from datetime import datetime
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.tools.builtin import (
    BraveSearchTool,
    CodeInterpreterTool,
    PhotogenTool,
    WolframAlphaTool,
)


def tool_breakdown(tools: List[ToolDefinition]) -> str:
    builtin_tools, custom_tools = [], []
    for dfn in tools:
        if isinstance(dfn.tool_name, BuiltinTool):
            builtin_tools.append(dfn)
        else:
            custom_tools.append(dfn)

    return builtin_tools, custom_tools


def prepare_messages_for_tools(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """This functions takes a ChatCompletionRequest and returns an augmented request.
    The request's messages are augmented to update the system message
    corresponding to the tool definitions provided in the request.
    """
    assert request.tool_choice == ToolChoice.auto, "Only `ToolChoice.auto` supported"

    existing_messages = request.messages

    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    builtin_tools, custom_tools = tool_breakdown(request.tools)

    messages = []
    content = ""
    if builtin_tools or custom_tools:
        content += "Environment: ipython\n"

    if builtin_tools:
        tool_str = ", ".join(
            [
                t.tool_name.value
                for t in builtin_tools
                if t.tool_name != BuiltinTool.code_interpreter
            ]
        )
        if tool_str:
            content += f"Tools: {tool_str}\n"

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")
    date_str = textwrap.dedent(
        f"""
        Cutting Knowledge Date: December 2023
        Today Date: {formatted_date}
        """
    )
    content += date_str.lstrip("\n")

    if existing_system_message:
        content += "\n"
        content += existing_system_message.content

    messages.append(SystemMessage(content=content))

    if custom_tools:
        if request.tool_prompt_format == ToolPromptFormat.function_tag:
            text = prompt_for_function_tag(custom_tools)
            messages.append(UserMessage(content=text))
        elif request.tool_prompt_format == ToolPromptFormat.json:
            text = prompt_for_json(custom_tools)
            messages.append(UserMessage(content=text))
        else:
            raise NotImplementedError(
                f"Tool prompt format {tool_prompt_format} is not supported"
            )

    messages += existing_messages
    request.messages = messages
    return request


def prompt_for_json(custom_tools: List[ToolDefinition]) -> str:
    tool_defs = "\n".join(
        translate_custom_tool_definition_to_json(t) for t in custom_tools
    )
    content = textwrap.dedent(
        """
        Answer the user's question by making use of the following functions if needed.
        If none of the function can be used, please say so.
        Here is a list of functions in JSON format:
        {tool_defs}

        Return function calls in JSON format.
        """
    )
    content = content.lstrip("\n").format(tool_defs=tool_defs)
    return content


def prompt_for_function_tag(custom_tools: List[ToolDefinition]) -> str:
    custom_tool_params = ""
    for t in custom_tools:
        custom_tool_params += get_instruction_string(t) + "\n"
        custom_tool_params += get_parameters_string(t) + "\n\n"

    content = textwrap.dedent(
        """
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
    )

    return content.lstrip("\n").format(custom_tool_params=custom_tool_params)


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

    return json.dumps(func_def, indent=4)
