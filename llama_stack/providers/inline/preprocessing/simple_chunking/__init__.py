# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import InclineSimpleChunkingConfig


async def get_provider_impl(
    config: InclineSimpleChunkingConfig,
    _deps,
):
    from .simple_chunking import InclineSimpleChunkingImpl

    impl = InclineSimpleChunkingImpl(config)
    await impl.initialize()
    return impl
