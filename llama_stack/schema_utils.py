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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# Borrowed from https://github.com/hunyadi/strong_typing/blob/master/strong_typing/core.py


class JsonObject:
    "Placeholder type for an unrestricted JSON object."


class JsonArray:
    "Placeholder type for an unrestricted JSON array."


# a JSON type with possible `null` values
JsonType = Union[
    None,
    bool,
    int,
    float,
    str,
    Dict[str, "JsonType"],
    List["JsonType"],
]

# a JSON type that cannot contain `null` values
StrictJsonType = Union[
    bool,
    int,
    float,
    str,
    Dict[str, "StrictJsonType"],
    List["StrictJsonType"],
]

# a meta-type that captures the object type in a JSON schema
Schema = Dict[str, JsonType]


T = TypeVar("T")


def register_schema(
    data_type: T,
    schema: Optional[Schema] = None,
    name: Optional[str] = None,
    examples: Optional[List[JsonType]] = None,
) -> T:
    """
    Associates a type with a JSON schema definition.

    :param data_type: The type to associate with a JSON schema.
    :param schema: The schema to associate the type with. Derived automatically if omitted.
    :param name: The name used for looking up the type. Determined automatically if omitted.
    :returns: The input type.
    """
    return data_type


def json_schema_type(
    cls: Optional[Type[T]] = None,
    *,
    schema: Optional[Schema] = None,
    examples: Optional[List[JsonType]] = None,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorator to add user-defined schema definition to a class."""

    def wrap(cls: Type[T]) -> Type[T]:
        return register_schema(cls, schema, examples=examples)

    # see if decorator is used as @json_schema_type or @json_schema_type()
    if cls is None:
        # called with parentheses
        return wrap
    else:
        # called as @json_schema_type without parentheses
        return wrap(cls)


register_schema(JsonObject, name="JsonObject")
register_schema(JsonArray, name="JsonArray")
register_schema(JsonType, name="JsonType")
register_schema(StrictJsonType, name="StrictJsonType")


@dataclass
class WebMethod:
    route: Optional[str] = None
    public: bool = False
    request_examples: Optional[List[Any]] = None
    response_examples: Optional[List[Any]] = None
    method: Optional[str] = None


def webmethod(
    route: Optional[str] = None,
    method: Optional[str] = None,
    public: Optional[bool] = False,
    request_examples: Optional[List[Any]] = None,
    response_examples: Optional[List[Any]] = None,
) -> Callable[[T], T]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    """

    def wrap(cls: T) -> T:
        cls.__webmethod__ = WebMethod(
            route=route,
            method=method,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
        )
        return cls

    return wrap
