# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

from .strong_typing.schema import json_schema_type, register_schema  # noqa: F401


@dataclass
class WebMethod:
    route: Optional[str] = None
    public: bool = False
    request_examples: Optional[List[Any]] = None
    response_examples: Optional[List[Any]] = None
    method: Optional[str] = None
    raw_bytes_request_body: Optional[bool] = False
    # A descriptive name of the corresponding span created by tracing
    descriptive_name: Optional[str] = None


T = TypeVar("T", bound=Callable[..., Any])


def webmethod(
    route: Optional[str] = None,
    method: Optional[str] = None,
    public: Optional[bool] = False,
    request_examples: Optional[List[Any]] = None,
    response_examples: Optional[List[Any]] = None,
    raw_bytes_request_body: Optional[bool] = False,
    descriptive_name: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    """

    def wrap(func: T) -> T:
        func.__webmethod__ = WebMethod(  # type: ignore
            route=route,
            method=method,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
            raw_bytes_request_body=raw_bytes_request_body,
            descriptive_name=descriptive_name,
        )
        return func

    return wrap
