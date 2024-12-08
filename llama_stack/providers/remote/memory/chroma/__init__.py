# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import ChromaConfig


async def get_adapter_impl(config: ChromaConfig, _deps):
    from .chroma import ChromaMemoryAdapter

    impl = ChromaMemoryAdapter(config)
    await impl.initialize()
    return impl
