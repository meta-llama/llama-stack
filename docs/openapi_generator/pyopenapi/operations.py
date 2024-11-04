# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import collections.abc
import enum
import inspect
import typing
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from termcolor import colored

from ..strong_typing.inspection import (
    get_signature,
    is_type_enum,
    is_type_optional,
    unwrap_optional_type,
)


def split_prefix(
    s: str, sep: str, prefix: Union[str, Iterable[str]]
) -> Tuple[Optional[str], str]:
    """
    Recognizes a prefix at the beginning of a string.

    :param s: The string to check.
    :param sep: A separator between (one of) the prefix(es) and the rest of the string.
    :param prefix: A string or a set of strings to identify as a prefix.
    :return: A tuple of the recognized prefix (if any) and the rest of the string excluding the separator (or the entire string).
    """

    if isinstance(prefix, str):
        if s.startswith(prefix + sep):
            return prefix, s[len(prefix) + len(sep) :]
        else:
            return None, s

    for p in prefix:
        if s.startswith(p + sep):
            return p, s[len(p) + len(sep) :]

    return None, s


def _get_annotation_type(annotation: Union[type, str], callable: Callable) -> type:
    "Maps a stringized reference to a type, as if using `from __future__ import annotations`."

    if isinstance(annotation, str):
        return eval(annotation, callable.__globals__)
    else:
        return annotation


class HTTPMethod(enum.Enum):
    "HTTP method used to invoke an endpoint operation."

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


OperationParameter = Tuple[str, type]


class ValidationError(TypeError):
    pass


@dataclass
class EndpointOperation:
    """
    Type information and metadata associated with an endpoint operation.

    "param defining_class: The most specific class that defines the endpoint operation.
    :param name: The short name of the endpoint operation.
    :param func_name: The name of the function to invoke when the operation is triggered.
    :param func_ref: The callable to invoke when the operation is triggered.
    :param route: A custom route string assigned to the operation.
    :param path_params: Parameters of the operation signature that are passed in the path component of the URL string.
    :param query_params: Parameters of the operation signature that are passed in the query string as `key=value` pairs.
    :param request_params: The parameter that corresponds to the data transmitted in the request body.
    :param event_type: The Python type of the data that is transmitted out-of-band (e.g. via websockets) while the operation is in progress.
    :param response_type: The Python type of the data that is transmitted in the response body.
    :param http_method: The HTTP method used to invoke the endpoint such as POST, GET or PUT.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take.
    :param response_examples: Sample responses that the operation might produce.
    """

    defining_class: type
    name: str
    func_name: str
    func_ref: Callable[..., Any]
    route: Optional[str]
    path_params: List[OperationParameter]
    query_params: List[OperationParameter]
    request_params: Optional[OperationParameter]
    event_type: Optional[type]
    response_type: type
    http_method: HTTPMethod
    public: bool
    request_examples: Optional[List[Any]] = None
    response_examples: Optional[List[Any]] = None

    def get_route(self) -> str:
        if self.route is not None:
            return self.route

        route_parts = ["", self.name]
        for param_name, _ in self.path_params:
            route_parts.append("{" + param_name + "}")
        return "/".join(route_parts)


class _FormatParameterExtractor:
    "A visitor to exract parameters in a format string."

    keys: List[str]

    def __init__(self) -> None:
        self.keys = []

    def __getitem__(self, key: str) -> None:
        self.keys.append(key)
        return None


def _get_route_parameters(route: str) -> List[str]:
    extractor = _FormatParameterExtractor()
    route.format_map(extractor)
    return extractor.keys


def _get_endpoint_functions(
    endpoint: type, prefixes: List[str]
) -> Iterator[Tuple[str, str, str, Callable]]:
    if not inspect.isclass(endpoint):
        raise ValueError(f"object is not a class type: {endpoint}")

    functions = inspect.getmembers(endpoint, inspect.isfunction)
    for func_name, func_ref in functions:
        webmethod = getattr(func_ref, "__webmethod__", None)
        if not webmethod:
            continue

        print(f"Processing {colored(func_name, 'white')}...")
        operation_name = func_name
        if operation_name.startswith("get_") or operation_name.endswith("/get"):
            prefix = "get"
        elif (
            operation_name.startswith("delete_")
            or operation_name.startswith("remove_")
            or operation_name.endswith("/delete")
            or operation_name.endswith("/remove")
        ):
            prefix = "delete"
        else:
            if webmethod.method == "GET":
                prefix = "get"
            elif webmethod.method == "DELETE":
                prefix = "delete"
            else:
                # by default everything else is a POST
                prefix = "post"

        yield prefix, operation_name, func_name, func_ref


def _get_defining_class(member_fn: str, derived_cls: type) -> type:
    "Find the class in which a member function is first defined in a class inheritance hierarchy."

    # iterate in reverse member resolution order to find most specific class first
    for cls in reversed(inspect.getmro(derived_cls)):
        for name, _ in inspect.getmembers(cls, inspect.isfunction):
            if name == member_fn:
                return cls

    raise ValidationError(
        f"cannot find defining class for {member_fn} in {derived_cls}"
    )


def get_endpoint_operations(
    endpoint: type, use_examples: bool = True
) -> List[EndpointOperation]:
    """
    Extracts a list of member functions in a class eligible for HTTP interface binding.

    These member functions are expected to have a signature like
    ```
    async def get_object(self, uuid: str, version: int) -> Object:
        ...
    ```
    where the prefix `get_` translates to an HTTP GET, `object` corresponds to the name of the endpoint operation,
    `uuid` and `version` are mapped to route path elements in "/object/{uuid}/{version}", and `Object` becomes
    the response payload type, transmitted as an object serialized to JSON.

    If the member function has a composite class type in the argument list, it becomes the request payload type,
    and the caller is expected to provide the data as serialized JSON in an HTTP POST request.

    :param endpoint: A class with member functions that can be mapped to an HTTP endpoint.
    :param use_examples: Whether to return examples associated with member functions.
    """

    result = []

    for prefix, operation_name, func_name, func_ref in _get_endpoint_functions(
        endpoint,
        [
            "create",
            "delete",
            "do",
            "get",
            "post",
            "put",
            "remove",
            "set",
            "update",
        ],
    ):
        # extract routing information from function metadata
        webmethod = getattr(func_ref, "__webmethod__", None)
        if webmethod is not None:
            route = webmethod.route
            route_params = _get_route_parameters(route) if route is not None else None
            public = webmethod.public
            request_examples = webmethod.request_examples
            response_examples = webmethod.response_examples
        else:
            route = None
            route_params = None
            public = False
            request_examples = None
            response_examples = None

        # inspect function signature for path and query parameters, and request/response payload type
        signature = get_signature(func_ref)

        path_params = []
        query_params = []
        request_params = []

        for param_name, parameter in signature.parameters.items():
            param_type = _get_annotation_type(parameter.annotation, func_ref)

            # omit "self" for instance methods
            if param_name == "self" and param_type is inspect.Parameter.empty:
                continue

            # check if all parameters have explicit type
            if parameter.annotation is inspect.Parameter.empty:
                raise ValidationError(
                    f"parameter '{param_name}' in function '{func_name}' has no type annotation"
                )

            if is_type_optional(param_type):
                inner_type: type = unwrap_optional_type(param_type)
            else:
                inner_type = param_type

            if prefix == "get" and (
                inner_type is bool
                or inner_type is int
                or inner_type is float
                or inner_type is str
                or inner_type is uuid.UUID
                or is_type_enum(inner_type)
            ):
                if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                    if route_params is not None and param_name not in route_params:
                        raise ValidationError(
                            f"positional parameter '{param_name}' absent from user-defined route '{route}' for function '{func_name}'"
                        )

                    # simple type maps to route path element, e.g. /study/{uuid}/{version}
                    path_params.append((param_name, param_type))
                else:
                    if route_params is not None and param_name in route_params:
                        raise ValidationError(
                            f"query parameter '{param_name}' found in user-defined route '{route}' for function '{func_name}'"
                        )

                    # simple type maps to key=value pair in query string
                    query_params.append((param_name, param_type))
            else:
                if route_params is not None and param_name in route_params:
                    raise ValidationError(
                        f"user-defined route '{route}' for function '{func_name}' has parameter '{param_name}' of composite type: {param_type}"
                    )

                request_params.append((param_name, param_type))

        # check if function has explicit return type
        if signature.return_annotation is inspect.Signature.empty:
            raise ValidationError(
                f"function '{func_name}' has no return type annotation"
            )

        return_type = _get_annotation_type(signature.return_annotation, func_ref)

        # operations that produce events are labeled as Generator[YieldType, SendType, ReturnType]
        # where YieldType is the event type, SendType is None, and ReturnType is the immediate response type to the request
        if typing.get_origin(return_type) is collections.abc.Generator:
            event_type, send_type, response_type = typing.get_args(return_type)
            if send_type is not type(None):
                raise ValidationError(
                    f"function '{func_name}' has a return type Generator[Y,S,R] and therefore looks like an event but has an explicit send type"
                )
        else:
            event_type = None

            def process_type(t):
                if typing.get_origin(t) is collections.abc.AsyncIterator:
                    # NOTE(ashwin): this is SSE and there is no way to represent it. either we make it a List
                    # or the item type. I am choosing it to be the latter
                    args = typing.get_args(t)
                    return args[0]
                elif typing.get_origin(t) is typing.Union:
                    types = [process_type(a) for a in typing.get_args(t)]
                    return typing._UnionGenericAlias(typing.Union, tuple(types))
                else:
                    return t

            response_type = process_type(return_type)

        # set HTTP request method based on type of request and presence of payload
        if not request_params:
            if prefix in ["delete", "remove"]:
                http_method = HTTPMethod.DELETE
            else:
                http_method = HTTPMethod.GET
        else:
            if prefix == "set":
                http_method = HTTPMethod.PUT
            elif prefix == "update":
                http_method = HTTPMethod.PATCH
            else:
                http_method = HTTPMethod.POST

        result.append(
            EndpointOperation(
                defining_class=_get_defining_class(func_name, endpoint),
                name=operation_name,
                func_name=func_name,
                func_ref=func_ref,
                route=route,
                path_params=path_params,
                query_params=query_params,
                request_params=request_params,
                event_type=event_type,
                response_type=response_type,
                http_method=http_method,
                public=public,
                request_examples=request_examples if use_examples else None,
                response_examples=response_examples if use_examples else None,
            )
        )

    if not result:
        raise ValidationError(f"no eligible endpoint operations in type {endpoint}")

    return result


def get_endpoint_events(endpoint: type) -> Dict[str, type]:
    results = {}

    for decl in typing.get_type_hints(endpoint).values():
        # check if signature is Callable[...]
        origin = typing.get_origin(decl)
        if origin is None or not issubclass(origin, Callable):  # type: ignore
            continue

        # check if signature is Callable[[...], Any]
        args = typing.get_args(decl)
        if len(args) != 2:
            continue
        params_type, return_type = args
        if not isinstance(params_type, list):
            continue

        # check if signature is Callable[[...], None]
        if not issubclass(return_type, type(None)):
            continue

        # check if signature is Callable[[EventType], None]
        if len(params_type) != 1:
            continue

        param_type = params_type[0]
        results[param_type.__name__] = param_type

    return results
