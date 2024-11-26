# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Type, TypeVar

from llama_stack.providers.utils.telemetry import tracing

T = TypeVar("T")


def trace_protocol(cls: Type[T]) -> Type[T]:
    """
    A class decorator that automatically traces all methods in a protocol/base class
    and its inheriting classes. Supports sync methods, async methods, and async generators.
    """

    def trace_method(method: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(method)
        is_async_gen = inspect.isasyncgenfunction(method)

        @wraps(method)
        async def async_gen_wrapper(
            self: Any, *args: Any, **kwargs: Any
        ) -> AsyncGenerator:
            class_name = self.__class__.__name__
            method_name = f"{class_name}.{method.__name__}"

            args_repr = [repr(arg) for arg in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            with tracing.span(
                f"{class_name}.{method_name}",
                {
                    "class": class_name,
                    "method": method_name,
                    "signature": signature,
                    "type": "async_generator",
                },
            ) as span:
                output = []
                try:
                    async for item in method(self, *args, **kwargs):
                        output.append(item)
                        yield item
                except Exception as e:
                    raise
                finally:
                    span.set_attribute("output", output)

        @wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            class_name = self.__class__.__name__
            method_name = f"{class_name}.{method.__name__}"

            args_repr = [repr(arg) for arg in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            with tracing.span(
                f"{class_name}.{method_name}",
                {
                    "class": class_name,
                    "method": method_name,
                    "signature": signature,
                    "type": "async",
                },
            ):
                try:
                    result = await method(self, *args, **kwargs)
                    return result
                except Exception as e:
                    raise

        @wraps(method)
        def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            class_name = self.__class__.__name__
            method_name = f"{class_name}.{method.__name__}"

            args_repr = [repr(arg) for arg in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            with tracing.span(
                f"{class_name}.{method_name}",
                {
                    "class": class_name,
                    "method": method_name,
                    "signature": signature,
                    "type": "sync",
                },
            ):
                try:
                    result = method(self, *args, **kwargs)
                    return result
                except Exception as e:
                    raise

        if is_async_gen:
            return async_gen_wrapper
        elif is_async:
            return async_wrapper
        else:
            return sync_wrapper

    # Trace all existing methods in the base class
    for name, method in vars(cls).items():
        if inspect.isfunction(method) and not name.startswith("__"):
            setattr(cls, name, trace_method(method))

    # Store the original __init_subclass__ if it exists
    original_init_subclass = getattr(cls, "__init_subclass__", None)

    # Define a new __init_subclass__ to handle child classes
    def __init_subclass__(cls_child, **kwargs):  # noqa: N807
        # Call original __init_subclass__ if it exists
        if original_init_subclass:
            original_init_subclass(**kwargs)

        # Trace all methods defined in the child class
        for name, method in vars(cls_child).items():
            if inspect.isfunction(method) and not name.startswith("__"):
                setattr(cls_child, name, trace_method(method))

    # Set the new __init_subclass__
    cls.__init_subclass__ = classmethod(__init_subclass__)

    return cls
