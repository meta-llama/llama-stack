# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Tuple

from llama_stack.distribution.datatypes import Api


async def get_router_impl(inner_impls: List[Tuple[str, Any]], deps: List[Api]):
    from .memory import MemoryRouterImpl

    impl = MemoryRouterImpl(inner_impls, deps)
    await impl.initialize()
    return impl
