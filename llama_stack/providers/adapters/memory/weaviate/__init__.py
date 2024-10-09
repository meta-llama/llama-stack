# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import WeaviateConfig, WeaviateRequestProviderData  # noqa: F401


async def get_adapter_impl(config: WeaviateConfig, _deps):
    from .weaviate import WeaviateMemoryAdapter

    impl = WeaviateMemoryAdapter(config)
    await impl.initialize()
    return impl
