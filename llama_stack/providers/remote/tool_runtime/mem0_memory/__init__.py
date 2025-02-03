# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.providers.datatypes import Api

from .config import Mem0ToolRuntimeConfig
from .memory import Mem0MemoryToolRuntimeImpl


async def get_adapter_impl(config: Mem0ToolRuntimeConfig, _deps):
    impl = Mem0MemoryToolRuntimeImpl(config)
    await impl.initialize()
    return impl
