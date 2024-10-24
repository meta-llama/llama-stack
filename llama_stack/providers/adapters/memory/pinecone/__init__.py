# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import PineconeConfig, PineconeRequestProviderData  # noqa: F401
from .pinecone import PineconeMemoryAdapter


async def get_adapter_impl(config: PineconeConfig, _deps):
    impl = PineconeMemoryAdapter(config)
    await impl.initialize()
    return impl
