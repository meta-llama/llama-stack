# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import PGVectorConfig


async def get_adapter_impl(config: PGVectorConfig, deps: Dict[Api, ProviderSpec]):
    from .pgvector import PGVectorMemoryAdapter

    impl = PGVectorMemoryAdapter(config, deps[Api.inference])
    await impl.initialize()
    return impl
