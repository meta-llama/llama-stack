# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import ChromaInlineImplConfig


async def get_provider_impl(config: ChromaInlineImplConfig, _deps):
    from llama_stack.providers.remote.memory.chroma.chroma import ChromaMemoryAdapter

    impl = ChromaMemoryAdapter(config)
    await impl.initialize()
    return impl
