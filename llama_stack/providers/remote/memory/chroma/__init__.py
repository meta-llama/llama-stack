# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.distribution.datatypes import RemoteProviderConfig
from llama_stack.providers.datatypes import Api, ProviderSpec


async def get_adapter_impl(config: RemoteProviderConfig, deps: Dict[Api, ProviderSpec]):
    from .chroma import ChromaMemoryAdapter

    impl = ChromaMemoryAdapter(config.url, deps[Api.inference])
    await impl.initialize()
    return impl
