# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import typing
import inspect
import os
from pathlib import Path
from typing import TextIO
from typing import Any, Dict, List, Optional, Protocol, Type, Union, get_type_hints, get_origin, get_args

from llama_stack.strong_typing.schema import object_to_json, StrictJsonType

from .generator import Generator
from .options import Options
from .specification import Document

THIS_DIR = Path(__file__).parent


class Specification:
    document: Document

    def __init__(self, endpoint: type, options: Options):
        generator = Generator(endpoint, options)
        self.document = generator.generate()

    def get_json(self) -> StrictJsonType:
        """
        Returns the OpenAPI specification as a Python data type (e.g. `dict` for an object, `list` for an array).

        The result can be serialized to a JSON string with `json.dump` or `json.dumps`.
        """

        json_doc = typing.cast(StrictJsonType, object_to_json(self.document))

        if isinstance(json_doc, dict):
            # rename vendor-specific properties
            tag_groups = json_doc.pop("tagGroups", None)
            if tag_groups:
                json_doc["x-tagGroups"] = tag_groups
            tags = json_doc.get("tags")
            if tags and isinstance(tags, list):
                for tag in tags:
                    if not isinstance(tag, dict):
                        continue

                    display_name = tag.pop("displayName", None)
                    if display_name:
                        tag["x-displayName"] = display_name

        return json_doc

    def get_json_string(self, pretty_print: bool = False) -> str:
        """
        Returns the OpenAPI specification as a JSON string.

        :param pretty_print: Whether to use line indents to beautify the output.
        """

        json_doc = self.get_json()
        if pretty_print:
            return json.dumps(
                json_doc, check_circular=False, ensure_ascii=False, indent=4
            )
        else:
            return json.dumps(
                json_doc,
                check_circular=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def write_json(self, f: TextIO, pretty_print: bool = False) -> None:
        """
        Writes the OpenAPI specification to a file as a JSON string.

        :param pretty_print: Whether to use line indents to beautify the output.
        """

        json_doc = self.get_json()
        if pretty_print:
            json.dump(
                json_doc,
                f,
                check_circular=False,
                ensure_ascii=False,
                indent=4,
            )
        else:
            json.dump(
                json_doc,
                f,
                check_circular=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def write_html(self, f: TextIO, pretty_print: bool = False) -> None:
        """
        Creates a stand-alone HTML page for the OpenAPI specification with ReDoc.

        :param pretty_print: Whether to use line indents to beautify the JSON string in the HTML file.
        """

        path = THIS_DIR / "template.html"
        with path.open(encoding="utf-8", errors="strict") as html_template_file:
            html_template = html_template_file.read()

        html = html_template.replace(
            "{ /* OPENAPI_SPECIFICATION */ }",
            self.get_json_string(pretty_print=pretty_print),
        )

        f.write(html)

def get_all_api_protocols() -> List[Type[Any]]:
    """Get all API protocol classes from the APIs directory."""
    protocols = []
    # Get the root directory of the project (where llama_stack is located)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    apis_dir = os.path.join(root_dir, "llama_stack", "apis")

    for root, _, files in os.walk(apis_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                module_path = os.path.relpath(os.path.join(root, file), apis_dir)
                module_name = f"llama_stack.apis.{module_path.replace(os.sep, '.').replace('.py', '')}"

                try:
                    module = __import__(module_name, fromlist=['*'])
                    for name, obj in inspect.getmembers(module):
                        # Check if it's a class that has methods with webmethod decorator
                        if inspect.isclass(obj):
                            methods = inspect.getmembers(obj, predicate=inspect.isfunction)
                            has_webmethod = any(hasattr(method, '__webmethod__') for _, method in methods)
                            if has_webmethod:
                                protocols.append(obj)
                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")

    return protocols

def is_optional_type(type_: Any) -> bool:
    """Check if a type is Optional."""
    origin = get_origin(type_)
    args = get_args(type_)
    return origin is Optional or (origin is Union and type(None) in args)


def validate_api_method_return_types() -> List[str]:
    """Validate that all API methods have proper return types."""
    errors = []
    protocols = get_all_api_protocols()

    for protocol in protocols:
        methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        for method_name, method in methods:
            if not hasattr(method, '__webmethod__'):
                continue

            # Only check GET methods
            if method.__webmethod__.method != "GET":
                continue

            hints = get_type_hints(method)

            if 'return' not in hints:
                errors.append(f"Method {protocol.__name__}.{method_name} has no return type annotation")
            else:
                return_type = hints['return']
                if is_optional_type(return_type):
                    errors.append(f"Method {protocol.__name__}.{method_name} returns Optional type")

    return errors
