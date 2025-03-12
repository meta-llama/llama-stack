# ruff: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import ast

from .tree_sitter import get_parser


def parse_java_function_call(source_code):
    if not source_code.endswith(";"):
        source_code += ";"  # Necessary for the parser not to register an error
    parser = get_parser("java")
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node

    if root_node.has_error:
        raise Exception("Error parsing java the source code.")

    def get_text(node):
        """Returns the text represented by the node."""
        return source_code[node.start_byte : node.end_byte]

    def traverse_node(node, nested=False):
        if node.type == "string_literal":
            if nested:
                return get_text(node)
            # Strip surrounding quotes from string literals
            return get_text(node)[1:-1]
        elif node.type == "character_literal":
            if nested:
                return get_text(node)
            # Strip surrounding single quotes from character literals
            return get_text(node)[1:-1]
        """Traverse the node to collect texts for complex structures."""
        if node.type in [
            "identifier",
            "class_literal",
            "type_identifier",
            "method_invocation",
        ]:
            return get_text(node)
        elif node.type == "array_creation_expression":
            # Handle array creation expression specifically
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")
            type_text = traverse_node(type_node, True)
            value_text = traverse_node(value_node, True)
            return f"new {type_text}[]{value_text}"
        elif node.type == "object_creation_expression":
            # Handle object creation expression specifically
            type_node = node.child_by_field_name("type")
            arguments_node = node.child_by_field_name("arguments")
            type_text = traverse_node(type_node, True)
            if arguments_node:
                # Process each argument carefully, avoiding unnecessary punctuation
                argument_texts = []
                for child in arguments_node.children:
                    if child.type not in [
                        ",",
                        "(",
                        ")",
                    ]:  # Exclude commas and parentheses
                        argument_text = traverse_node(child, True)
                        argument_texts.append(argument_text)
                arguments_text = ", ".join(argument_texts)
                return f"new {type_text}({arguments_text})"
            else:
                return f"new {type_text}()"
        elif node.type == "set":
            # Handling sets specifically
            items = [traverse_node(n, True) for n in node.children if n.type not in [",", "set"]]
            return "{" + ", ".join(items) + "}"

        elif node.child_count > 0:
            return "".join(traverse_node(child, True) for child in node.children)
        else:
            return get_text(node)

    def extract_arguments(args_node):
        arguments = {}
        for child in args_node.children:
            if child.type == "assignment_expression":
                # For named parameters
                name_node, value_node = child.children[0], child.children[2]
                name = get_text(name_node)
                value = traverse_node(value_node)
                if name in arguments:
                    if not isinstance(arguments[name], list):
                        arguments[name] = [arguments[name]]
                    arguments[name].append(value)
                else:
                    arguments[name] = value
                # arguments.append({'name': name, 'value': value})
            elif child.type in ["identifier", "class_literal", "set"]:
                # For unnamed parameters and handling sets
                value = traverse_node(child)
                if None in arguments:
                    if not isinstance(arguments[None], list):
                        arguments[None] = [arguments[None]]
                    arguments[None].append(value)
                else:
                    arguments[None] = value
        return arguments

    def traverse(node):
        if node.type == "method_invocation":
            # Extract the function name and its arguments
            method_name = get_text(node.child_by_field_name("name"))
            class_name_node = node.child_by_field_name("object")
            if class_name_node:
                class_name = get_text(class_name_node)
                function_name = f"{class_name}.{method_name}"
            else:
                function_name = method_name
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node:
                arguments = extract_arguments(arguments_node)
                for key, value in arguments.items():
                    if isinstance(value, list):
                        raise Exception("Error: Multiple arguments with the same name are not supported.")
                return [{function_name: arguments}]

        else:
            for child in node.children:
                result = traverse(child)
                if result:
                    return result

    result = traverse(root_node)
    return result if result else {}


def parse_javascript_function_call(source_code):
    if not source_code.endswith(";"):
        source_code += ";"  # Necessary for the parser not to register an error
    parser = get_parser("javascript")
    # Parse the source code
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    if root_node.has_error:
        raise Exception("Error js parsing the source code.")

    # Function to recursively extract argument details
    def extract_arguments(node):
        args = {}
        for child in node.children:
            if child.type == "assignment_expression":
                # Extract left (name) and right (value) parts of the assignment
                name = child.children[0].text.decode("utf-8")
                value = child.children[2].text.decode("utf-8")
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]  # Trim the quotation marks
                if name in args:
                    if not isinstance(args[name], list):
                        args[name] = [args[name]]
                    args[name].append(value)
                else:
                    args[name] = value

            elif child.type == "identifier" or child.type == "true":
                # Handle non-named arguments and boolean values
                value = child.text.decode("utf-8")
                if None in args:
                    if not isinstance(args[None], list):
                        args[None] = [args[None]]
                    args[None].append(value)
                else:
                    args[None] = value
        return args

    # Find the function call and extract its name and arguments
    if root_node.type == "program":
        for child in root_node.children:
            if child.type == "expression_statement":
                for sub_child in child.children:
                    if sub_child.type == "call_expression":
                        function_name = sub_child.children[0].text.decode("utf8")
                        arguments_node = sub_child.children[1]
                        parameters = extract_arguments(arguments_node)
                        for key, value in parameters.items():
                            if isinstance(value, list):
                                raise Exception("Error: Multiple arguments with the same name are not supported.")
                        result = [{function_name: parameters}]
                        return result


def ast_parse(input_str, language="Python"):
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                extracted.append(resolve_ast_call(elem))
        return extracted
    elif language == "Java":
        return parse_java_function_call(input_str[1:-1])  # Remove the [ and ] from the string
    elif language == "JavaScript":
        return parse_javascript_function_call(input_str[1:-1])
    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    # Parse when args are simply passed as an unnamed dictionary arg
    for arg in elem.args:
        if isinstance(arg, ast.Dict):
            for key, value in zip(arg.keys, arg.values):
                if isinstance(key, ast.Constant):
                    arg_name = key.value
                output = resolve_ast_by_type(value)
                args_dict[arg_name] = output
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    elif isinstance(value, ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(value, ast.BinOp):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def decode_ast(result, language="Python"):
    func = result
    func = func.replace("\n", "")  # remove new line characters
    if not func.startswith("["):
        func = "[" + func
    if not func.endswith("]"):
        func = func + "]"
    decoded_output = ast_parse(func, language)
    return decoded_output


def decode_execute(result):
    func = result
    func = func.replace("\n", "")  # remove new line characters
    if not func.startswith("["):
        func = "[" + func
    if not func.endswith("]"):
        func = func + "]"
    decode_output = ast_parse(func)
    execution_list = []
    for function_call in decode_output:
        for key, value in function_call.items():
            execution_list.append(f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})")
    return execution_list
