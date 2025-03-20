# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.
import ast
import json
import re
from typing import Optional, Tuple

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import BuiltinTool, RecursiveType, ToolCall, ToolPromptFormat

logger = get_logger(name=__name__, category="inference")

BUILTIN_TOOL_PATTERN = r'\b(?P<tool_name>\w+)\.call\(query="(?P<query>[^"]*)"\)'
CUSTOM_TOOL_CALL_PATTERN = re.compile(r"<function=(?P<function_name>[^}]+)>(?P<args>{.*?})")


def is_json(s):
    try:
        parsed = json.loads(s)
        # Return True for valid objects and not for ints, strings, etc
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False
    return True


def is_valid_python_list(input_string):
    """Check if the input string is a valid Python list of function calls"""
    try:
        # Try to parse the string
        tree = ast.parse(input_string)

        # Check if it's a single expression
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return False

        # Check if the expression is a list
        expr = tree.body[0].value
        if not isinstance(expr, ast.List):
            return False

        # Check if the list is empty
        if len(expr.elts) == 0:
            return False

        # Check if all elements in the list are function calls
        for element in expr.elts:
            if not isinstance(element, ast.Call):
                return False

            # Check if the function call has a valid name
            if not isinstance(element.func, ast.Name):
                return False

            # Check if all arguments are keyword arguments
            if element.args or not all(isinstance(arg, ast.keyword) for arg in element.keywords):
                return False

        return True

    except SyntaxError:
        # If parsing fails, it's not a valid Python expression
        return False


def parse_python_list_for_function_calls(input_string):
    """
    Parse a Python list of function calls and
    return a list of tuples containing the function name and arguments
    """
    # Parse the string into an AST
    tree = ast.parse(input_string)

    # Ensure the input is a list
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.List):
        raise ValueError("Input must be a list of function calls")

    result = []

    # Iterate through each function call in the list
    for node in tree.body[0].value.elts:
        if isinstance(node, ast.Call):
            function_name = node.func.id
            function_args = {}

            # Extract keyword arguments
            for keyword in node.keywords:
                try:
                    function_args[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError as e:
                    logger.error(
                        f"Error parsing tool call argument '{keyword.arg}': {e}, full input string: '{input_string}'"
                    )
                    raise ValueError(
                        f"Error parsing tool call argument '{keyword.arg}', full input string: '{input_string}'"
                    ) from e

            result.append((function_name, function_args))

    return result


class ToolUtils:
    @staticmethod
    def is_builtin_tool_call(message_body: str) -> bool:
        match = re.search(ToolUtils.BUILTIN_TOOL_PATTERN, message_body)
        return match is not None

    @staticmethod
    def maybe_extract_builtin_tool_call(message_body: str) -> Optional[Tuple[str, str]]:
        # Find the first match in the text
        match = re.search(BUILTIN_TOOL_PATTERN, message_body)

        # Check if a match is found and return it
        if match:
            tool_name = match.group("tool_name")
            query = match.group("query")
            return tool_name, query
        else:
            return None

    @staticmethod
    def maybe_extract_custom_tool_call(message_body: str) -> Optional[Tuple[str, str]]:
        # NOTE: Custom function too calls are still experimental
        # Sometimes, response is of the form
        # {"type": "function", "name": "function_name", "parameters": {...}
        # and some times
        # <function=function_name>(parameters)</function>

        # Find the first match in the text
        match = re.search(CUSTOM_TOOL_CALL_PATTERN, message_body)
        if match:
            tool_name = match.group("function_name")
            query = match.group("args")
            try:
                return tool_name, json.loads(query.replace("'", '"'))
            except Exception as e:
                print("Exception while parsing json query for custom tool call", query, e)
                return None
        elif is_json(message_body):
            response = json.loads(message_body)
            if ("type" in response and response["type"] == "function") or ("name" in response):
                function_name = response["name"]
                args = response["parameters"]
                return function_name, args
            else:
                return None
        elif is_valid_python_list(message_body):
            res = parse_python_list_for_function_calls(message_body)
            # FIXME: Enable multiple tool calls
            return res[0]
        else:
            return None

    @staticmethod
    def encode_tool_call(t: ToolCall, tool_prompt_format: ToolPromptFormat) -> str:
        if t.tool_name == BuiltinTool.brave_search:
            q = t.arguments["query"]
            return f'brave_search.call(query="{q}")'
        elif t.tool_name == BuiltinTool.wolfram_alpha:
            q = t.arguments["query"]
            return f'wolfram_alpha.call(query="{q}")'
        elif t.tool_name == BuiltinTool.photogen:
            q = t.arguments["query"]
            return f'photogen.call(query="{q}")'
        elif t.tool_name == BuiltinTool.code_interpreter:
            return t.arguments["code"]
        else:
            fname = t.tool_name

            if tool_prompt_format == ToolPromptFormat.json:
                return json.dumps(
                    {
                        "type": "function",
                        "name": fname,
                        "parameters": t.arguments,
                    }
                )
            elif tool_prompt_format == ToolPromptFormat.function_tag:
                args = json.dumps(t.arguments)
                return f"<function={fname}>{args}</function>"

            elif tool_prompt_format == ToolPromptFormat.python_list:

                def format_value(value: RecursiveType) -> str:
                    if isinstance(value, str):
                        return f'"{value}"'
                    elif isinstance(value, (int, float, bool)) or value is None:
                        return str(value)
                    elif isinstance(value, list):
                        return f"[{', '.join(format_value(v) for v in value)}]"
                    elif isinstance(value, dict):
                        return f"{{{', '.join(f'{k}={format_value(v)}' for k, v in value.items())}}}"
                    else:
                        raise ValueError(f"Unsupported type: {type(value)}")

                args_str = ", ".join(f"{k}={format_value(v)}" for k, v in t.arguments.items())
                return f"[{fname}({args_str})]"
            else:
                raise ValueError(f"Unsupported tool prompt format: {tool_prompt_format}")
