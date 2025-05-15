# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api, ProviderContext

from .config import MilvusVectorIOConfig


async def get_provider_impl(context: ProviderContext, config: MilvusVectorIOConfig, deps: dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter

    # Pass config directly since MilvusVectorIOAdapter doesn't accept context
    impl = MilvusVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
