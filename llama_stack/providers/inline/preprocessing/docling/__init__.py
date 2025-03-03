# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import InlineDoclingConfig


async def get_provider_impl(
    config: InlineDoclingConfig,
    _deps,
):
    from .docling import InclineDoclingPreprocessorImpl

    impl = InclineDoclingPreprocessorImpl(config)
    await impl.initialize()
    return impl
