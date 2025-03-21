# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.providers.datatypes import Api

from .config import RagToolRuntimeConfig


async def get_provider_impl(config: RagToolRuntimeConfig, deps: Dict[Api, Any]):
    from .memory import MemoryToolRuntimeImpl

    impl = MemoryToolRuntimeImpl(config, deps[Api.vector_io], deps[Api.inference])
    await impl.initialize()
    return impl
