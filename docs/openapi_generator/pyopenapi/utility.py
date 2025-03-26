# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import typing
import inspect
from pathlib import Path
from typing import TextIO
from typing import Any, List, Optional, Union, get_type_hints, get_origin, get_args

from llama_stack.strong_typing.schema import object_to_json, StrictJsonType
from llama_stack.distribution.resolver import api_protocol_map

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

def is_optional_type(type_: Any) -> bool:
    """Check if a type is Optional."""
    origin = get_origin(type_)
    args = get_args(type_)
    return origin is Optional or (origin is Union and type(None) in args)


def _validate_api_method_return_type(method) -> str | None:
    hints = get_type_hints(method)

    if 'return' not in hints:
        return "has no return type annotation"

    return_type = hints['return']
    if is_optional_type(return_type):
        return "returns Optional type"


def _validate_api_delete_method_returns_none(method) -> str | None:
    hints = get_type_hints(method)

    if 'return' not in hints:
        return "has no return type annotation"

    return_type = hints['return']
    if return_type is not None and return_type is not type(None):
        return "does not return None"


def _validate_list_parameters_contain_data(method) -> str | None:
    hints = get_type_hints(method)

    if 'return' not in hints:
        return "has no return type annotation"

    return_type = hints['return']
    if not inspect.isclass(return_type):
        return

    if not return_type.__name__.startswith('List'):
        return

    if 'data' not in return_type.model_fields:
        return "does not have data attribute"


_VALIDATORS = {
    "GET": [
        _validate_api_method_return_type,
        _validate_list_parameters_contain_data,
    ],
    "DELETE": [
        _validate_api_delete_method_returns_none,
    ],
}


def _get_methods_by_type(protocol, method_type: str):
    members = inspect.getmembers(protocol, predicate=inspect.isfunction)
    return {
        method_name: method
        for method_name, method in members
        if (webmethod := getattr(method, '__webmethod__', None))
        if webmethod and webmethod.method == method_type
    }


def validate_api() -> List[str]:
    """Validate the API protocols."""
    errors = []
    protocols = api_protocol_map()

    for target, validators in _VALIDATORS.items():
        for protocol_name, protocol in protocols.items():
            for validator in validators:
                for method_name, method in _get_methods_by_type(protocol, target).items():
                    err = validator(method)
                    if err:
                        errors.append(f"Method {protocol_name}.{method_name} {err}")

    return errors
