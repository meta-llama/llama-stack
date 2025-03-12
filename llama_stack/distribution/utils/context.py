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
    Wraps an async generator to preserve context variables across iterations.
    This is needed because we start a new asyncio event loop for each streaming request,
    and we need to preserve the context across the event loop boundary.
    """

    async def wrapper() -> AsyncGenerator[T, None]:
        while True:
            try:
                item = await gen.__anext__()
                context_values = {context_var.name: context_var.get() for context_var in context_vars}
                yield item
                for context_var in context_vars:
                    _ = context_var.set(context_values[context_var.name])
            except StopAsyncIteration:
                break

    return wrapper()
