# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect

import json
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, get_args, get_origin, Type, Union

import httpx
from pydantic import BaseModel, parse_obj_as
from termcolor import cprint

from llama_stack.providers.datatypes import RemoteProviderConfig

_CLIENT_CLASSES = {}


async def get_client_impl(
    protocol, additional_protocol, config: RemoteProviderConfig, _deps: Any
):
    client_class = create_api_client_class(protocol, additional_protocol)
    impl = client_class(config.url)
    await impl.initialize()
    return impl


def create_api_client_class(protocol, additional_protocol) -> Type:
    if protocol in _CLIENT_CLASSES:
        return _CLIENT_CLASSES[protocol]

    protocols = [protocol, additional_protocol] if additional_protocol else [protocol]

    class APIClient:
        def __init__(self, base_url: str):
            print(f"({protocol.__name__}) Connecting to {base_url}")
            self.base_url = base_url.rstrip("/")
            self.routes = {}

            # Store routes for this protocol
            for p in protocols:
                for name, method in inspect.getmembers(p):
                    if hasattr(method, "__webmethod__"):
                        sig = inspect.signature(method)
                        self.routes[name] = (method.__webmethod__, sig)

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

        async def __acall__(self, method_name: str, *args, **kwargs) -> Any:
            assert method_name in self.routes, f"Unknown endpoint: {method_name}"

            # TODO: make this more precise, same thing needs to happen in server.py
            is_streaming = kwargs.get("stream", False)
            if is_streaming:
                return self._call_streaming(method_name, *args, **kwargs)
            else:
                return await self._call_non_streaming(method_name, *args, **kwargs)

        async def _call_non_streaming(self, method_name: str, *args, **kwargs) -> Any:
            _, sig = self.routes[method_name]

            if sig.return_annotation is None:
                return_type = None
            else:
                return_type = extract_non_async_iterator_type(sig.return_annotation)
                assert (
                    return_type
                ), f"Could not extract return type for {sig.return_annotation}"

            async with httpx.AsyncClient() as client:
                params = self.httpx_request_params(method_name, *args, **kwargs)
                response = await client.request(**params)
                response.raise_for_status()

                j = response.json()
                if j is None:
                    return None
                # print(f"({protocol.__name__}) Returning {j}, type {return_type}")
                return parse_obj_as(return_type, j)

        async def _call_streaming(self, method_name: str, *args, **kwargs) -> Any:
            webmethod, sig = self.routes[method_name]

            return_type = extract_async_iterator_type(sig.return_annotation)
            assert (
                return_type
            ), f"Could not extract return type for {sig.return_annotation}"

            async with httpx.AsyncClient() as client:
                params = self.httpx_request_params(method_name, *args, **kwargs)
                async with client.stream(**params) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[len("data: ") :]
                            try:
                                data = json.loads(data)
                                if "error" in data:
                                    cprint(data, "red")
                                    continue

                                yield parse_obj_as(return_type, data)
                            except Exception as e:
                                print(f"Error with parsing or validation: {e}")
                                print(data)

        def httpx_request_params(self, method_name: str, *args, **kwargs) -> dict:
            webmethod, sig = self.routes[method_name]

            parameters = list(sig.parameters.values())[1:]  # skip `self`
            for i, param in enumerate(parameters):
                if i >= len(args):
                    break
                kwargs[param.name] = args[i]

            url = f"{self.base_url}{webmethod.route}"

            def convert(value):
                if isinstance(value, list):
                    return [convert(v) for v in value]
                elif isinstance(value, dict):
                    return {k: convert(v) for k, v in value.items()}
                elif isinstance(value, BaseModel):
                    return json.loads(value.model_dump_json())
                elif isinstance(value, Enum):
                    return value.value
                else:
                    return value

            params = {}
            data = {}
            if webmethod.method == "GET":
                params.update(kwargs)
            else:
                data.update(convert(kwargs))

            return dict(
                method=webmethod.method or "POST",
                url=url,
                headers={"Content-Type": "application/json"},
                params=params,
                json=data,
                timeout=30,
            )

    # Add protocol methods to the wrapper
    for p in protocols:
        for name, method in inspect.getmembers(p):
            if hasattr(method, "__webmethod__"):

                async def method_impl(self, *args, method_name=name, **kwargs):
                    return await self.__acall__(method_name, *args, **kwargs)

                method_impl.__name__ = name
                method_impl.__qualname__ = f"APIClient.{name}"
                method_impl.__signature__ = inspect.signature(method)
                setattr(APIClient, name, method_impl)

    # Name the class after the protocol
    APIClient.__name__ = f"{protocol.__name__}Client"
    _CLIENT_CLASSES[protocol] = APIClient
    return APIClient


# not quite general these methods are
def extract_non_async_iterator_type(type_hint):
    if get_origin(type_hint) is Union:
        args = get_args(type_hint)
        for arg in args:
            if not issubclass(get_origin(arg) or arg, AsyncIterator):
                return arg
    return type_hint


def extract_async_iterator_type(type_hint):
    if get_origin(type_hint) is Union:
        args = get_args(type_hint)
        for arg in args:
            if issubclass(get_origin(arg) or arg, AsyncIterator):
                inner_args = get_args(arg)
                return inner_args[0]
    return None


async def example(model: str = None):
    from llama_stack.apis.inference import Inference, UserMessage  # noqa: F403
    from llama_stack.apis.inference.event_logger import EventLogger

    client_class = create_api_client_class(Inference)
    client = client_class("http://localhost:5003")

    if not model:
        model = "Llama3.2-3B-Instruct"

    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon"
    )
    cprint(f"User>{message.content}", "green")

    stream = True
    iterator = await client.chat_completion(
        model=model,
        messages=[message],
        stream=stream,
    )

    async for log in EventLogger().log(iterator):
        log.print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(example())
