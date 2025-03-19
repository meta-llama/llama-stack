# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import QdrantVectorIOConfig


async def get_adapter_impl(config: QdrantVectorIOConfig, deps: Dict[Api, ProviderSpec]):
    from llama_stack.providers.remote.vector_io.qdrant.qdrant import QdrantVectorIOAdapter

    impl = QdrantVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
