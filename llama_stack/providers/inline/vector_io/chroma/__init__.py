# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api, ProviderContext

from .config import ChromaVectorIOConfig


async def get_provider_impl(context: ProviderContext, config: ChromaVectorIOConfig, deps: dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.chroma.chroma import (
        ChromaVectorIOAdapter,
    )

    # Pass config directly since ChromaVectorIOAdapter doesn't accept context
    impl = ChromaVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
