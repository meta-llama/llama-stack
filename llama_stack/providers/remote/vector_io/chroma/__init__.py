# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import ChromaVectorIOConfig


async def get_adapter_impl(config: ChromaVectorIOConfig, deps: Dict[Api, ProviderSpec]):
    from .chroma import ChromaVectorIOAdapter

    impl = ChromaVectorIOAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
