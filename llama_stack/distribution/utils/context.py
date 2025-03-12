# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from contextvars import ContextVar
from typing import AsyncGenerator, List, TypeVar

T = TypeVar("T")


def preserve_contexts_async_generator(
    gen: AsyncGenerator[T, None], context_vars: List[ContextVar]
) -> AsyncGenerator[T, None]:
    """
    Wraps an async generator to preserve both tracing and headers context variables across iterations.
    This is needed because we start a new asyncio event loop for each request, and we need to preserve the context
    across the event loop boundary.
    """
    context_values = [context_var.get() for context_var in context_vars]

    async def wrapper():
        while True:
            for context_var, context_value in zip(context_vars, context_values, strict=False):
                _ = context_var.set(context_value)
            try:
                item = await gen.__anext__()
                yield item
            except StopAsyncIteration:
                break

    return wrapper()
