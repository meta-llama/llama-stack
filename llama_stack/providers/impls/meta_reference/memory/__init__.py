# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import FaissImplConfig


async def get_provider_impl(config: FaissImplConfig, _deps):
    from .faiss import FaissMemoryImpl

    assert isinstance(
        config, FaissImplConfig
    ), f"Unexpected config type: {type(config)}"

    impl = FaissMemoryImpl(config)
    await impl.initialize()
    return impl
